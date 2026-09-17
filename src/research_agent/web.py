from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import ResearchAgent
from .cli import _demo_actions
from .model import GroqPlanner, LibraPlanner, OpenAIPlanner, ScriptedPlanner
from .metrics import render_metrics
from .tools import HttpTools, LocalCorpusTools


RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()
ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / ".runs" / "ui"
LOGGER = logging.getLogger(__name__)
RUN_ID_PATTERN = re.compile(r"[a-f0-9]{10}")
PROVIDERS = {"openai", "groq", "libra"}


def _safe_run_dir(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid saved run ID")
    run_dir = (RUNS_ROOT / run_id).resolve()
    if run_dir.parent != RUNS_ROOT.resolve():
        raise ValueError("saved run is outside the run directory")
    return run_dir


def _read_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}")
    return value


def _read_run_config(run_dir: Path) -> dict:
    metadata = _read_json_object(run_dir / "run.json", "run metadata")
    mode = metadata.get("mode")
    provider = metadata.get("provider")
    model = metadata.get("model")
    if metadata.get("id") != run_dir.name:
        raise ValueError("run metadata ID does not match its directory")
    if mode not in {"demo", "live"} or provider not in PROVIDERS or not isinstance(model, str) or not model.strip():
        raise ValueError("invalid run metadata")
    return {"mode": mode, "provider": provider, "model": model.strip()}


def _write_run_config(run_dir: Path, run_id: str, mode: str, provider: str, model: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    created_at = time.time()
    if path.exists():
        try:
            created_at = float(_read_json_object(path, "run metadata").get("created_at", created_at))
        except (TypeError, ValueError):
            pass
    payload = {
        "version": 1,
        "id": run_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "created_at": created_at,
        "updated_at": time.time(),
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _saved_runs() -> list[dict]:
    if not RUNS_ROOT.exists():
        return []
    with RUNS_LOCK:
        active_ids = {
            run_id for run_id, run in RUNS.items() if run.get("status") in {"starting", "running"}
        }
    saved = []
    for checkpoint_path in RUNS_ROOT.glob("*/checkpoint.json"):
        run_id = checkpoint_path.parent.name
        if run_id in active_ids or not RUN_ID_PATTERN.fullmatch(run_id):
            continue
        try:
            run_dir = _safe_run_dir(run_id)
            if run_dir != checkpoint_path.parent.resolve():
                continue
            checkpoint = _read_json_object(run_dir / "checkpoint.json", "checkpoint")
            config = _read_run_config(run_dir)
            status = checkpoint.get("status")
            goal = checkpoint.get("goal")
            step = checkpoint.get("step")
            if status not in {"running", "failed", "budget_exhausted"}:
                continue
            if not isinstance(goal, str) or not goal.strip() or not isinstance(step, int) or step < 0:
                continue
            saved.append(
                {
                    "id": run_id,
                    "goal": goal,
                    "status": status,
                    "steps": step,
                    "minimum_max_steps": step + 1,
                    "updated_at": (run_dir / "checkpoint.json").stat().st_mtime,
                    **config,
                }
            )
        except (OSError, ValueError):
            continue
    return sorted(saved, key=lambda item: item["updated_at"], reverse=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "ResearchAgentUI/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send(HTTPStatus.OK, HTML, "text/html; charset=utf-8")
        if path == "/metrics":
            return self._send(HTTPStatus.OK, render_metrics(RUNS_ROOT), "text/plain; version=0.0.4; charset=utf-8")
        if path == "/api/runs":
            return self._json(HTTPStatus.OK, {"runs": _saved_runs()})
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                run = dict(RUNS.get(run_id, {}))
            if not run:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
            trace_path = Path(run["trace_path"])
            run["events"] = _read_events(trace_path)
            return self._json(HTTPStatus.OK, run)
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/runs":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            resume = body.get("resume", False) is True
            try:
                max_steps = int(body.get("max_steps", 12))
            except (TypeError, ValueError) as exc:
                raise ValueError("Maximum steps must be an integer.") from exc
            if not 1 <= max_steps <= 100:
                raise ValueError("Maximum steps must be between 1 and 100.")

            starting_step = 0
            if resume:
                run_id = str(body.get("run_id", "")).strip()
                run_dir = _safe_run_dir(run_id)
                checkpoint = _read_json_object(run_dir / "checkpoint.json", "checkpoint")
                config = _read_run_config(run_dir)
                checkpoint_status = checkpoint.get("status")
                raw_goal = checkpoint.get("goal")
                checkpoint_step = checkpoint.get("step")
                if checkpoint_status == "complete":
                    raise ValueError("A completed run cannot be resumed.")
                if checkpoint_status not in {"running", "failed", "budget_exhausted"}:
                    raise ValueError("The saved checkpoint has an invalid status.")
                if not isinstance(raw_goal, str) or not raw_goal.strip():
                    raise ValueError("The saved checkpoint has no research goal.")
                goal = raw_goal.strip()
                if not isinstance(checkpoint_step, int) or max_steps <= checkpoint_step:
                    raise ValueError(f"Maximum steps must be greater than the saved step ({checkpoint_step}).")
                starting_step = checkpoint_step
                with RUNS_LOCK:
                    if RUNS.get(run_id, {}).get("status") in {"starting", "running"}:
                        raise ValueError("This run is already active.")
                demo = config["mode"] == "demo"
                provider = config["provider"]
                model = config["model"]
            else:
                run_id = uuid.uuid4().hex[:10]
                run_dir = _safe_run_dir(run_id)
                goal = str(body.get("goal", "")).strip()
                demo = body.get("demo", False) is True
                provider = str(body.get("provider", "openai")).strip().lower()
                model = str(body.get("model", "gpt-5-mini")).strip()
                if not goal:
                    raise ValueError("Please enter a research goal.")
                if provider not in PROVIDERS:
                    raise ValueError("Provider must be OpenAI, Groq, or Libra.")
                if not model:
                    raise ValueError("Model is required.")

            key_env = {
                "openai": "OPENAI_API_KEY",
                "groq": "GROQ_API_KEY",
                "libra": "LIBRA_INTERVIEW_API_KEY",
            }[provider]
            api_key = str(body.get("api_key", "")).strip() or os.environ.get(key_env, "")
            if not demo and not api_key:
                raise ValueError(f"Enter a {provider.title()} API key or select offline demo.")
        except ValueError as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        try:
            _write_run_config(run_dir, run_id, "demo" if demo else "live", provider, model)
        except OSError:
            return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Could not persist run metadata."})
        with RUNS_LOCK:
            RUNS[run_id] = {
                "id": run_id,
                "goal": goal,
                "mode": "demo" if demo else "live",
                "provider": provider,
                "model": model,
                "resume": resume,
                "max_steps": max_steps,
                "status": "starting",
                "answer": None,
                "error": None,
                "steps": starting_step,
                "provider_retries": 0,
                "tool_retries": 0,
                "validation_failures": 0,
                "quality_checks": {},
                "trace_path": str(run_dir / "trace.jsonl"),
            }
        threading.Thread(
            target=_run_agent,
            args=(run_id, goal, demo, provider, model, run_dir, api_key, resume, max_steps),
            daemon=True,
        ).start()
        self._json(HTTPStatus.ACCEPTED, {"id": run_id})

    def log_message(self, format: str, *args) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, content: str | bytes, content_type: str) -> None:
        data = content.encode() if isinstance(content, str) else content
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def _run_agent(
    run_id: str,
    goal: str,
    demo: bool,
    provider: str,
    model: str,
    run_dir: Path,
    api_key: str,
    resume: bool = False,
    max_steps: int = 12,
) -> None:
    with RUNS_LOCK:
        RUNS[run_id]["status"] = "running"
    LOGGER.info("run_started run_id=%s mode=%s provider=%s", run_id, "demo" if demo else "live", provider)
    try:
        if demo:
            completed_steps = 0
            if resume:
                completed_steps = int(_read_json_object(run_dir / "checkpoint.json", "checkpoint").get("step", 0))
            planner = ScriptedPlanner(_demo_actions()[completed_steps:])
            tools = LocalCorpusTools(ROOT / "evals" / "corpus")
        else:
            if provider == "groq":
                planner = GroqPlanner(model, api_key=api_key)
            elif provider == "libra":
                planner = LibraPlanner(model, api_key=api_key)
            else:
                planner = OpenAIPlanner(model, api_key=api_key)
            tools = HttpTools()
        state = ResearchAgent(planner, tools, run_dir, max_steps=max_steps).run(goal, resume=resume)
        with RUNS_LOCK:
            RUNS[run_id].update(
                status=state.status,
                answer=state.final_answer,
                steps=state.step,
                provider_retries=state.provider_retries,
                tool_retries=state.tool_retries,
                validation_failures=state.validation_failures,
                quality_checks=getattr(state, "quality_checks", {}),
            )
        LOGGER.info("run_finished run_id=%s status=%s steps=%s", run_id, state.status, state.step)
    except Exception as exc:
        safe_error = ResearchAgent._redact_text(str(exc))
        with RUNS_LOCK:
            RUNS[run_id].update(status="failed", error=f"{type(exc).__name__}: {safe_error}")
        # Deliberately avoid logging.exception(): provider exceptions can contain credentials.
        LOGGER.error("run_failed run_id=%s error_type=%s", run_id, type(exc).__name__)  # NOSONAR


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal research-agent UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    # Plain HTTP is intentional for the loopback-only development UI. Docker also publishes to 127.0.0.1.
    url = f"http://{args.host}:{args.port}"  # NOSONAR
    print(f"Research Agent UI: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resilient Research Agent</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--line:#29334d;--text:#edf2ff;--muted:#9aa8c7;--accent:#7dd3fc;--good:#86efac;--bad:#fca5a5}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 15% 0,#172554 0,transparent 32%),var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}.wrap{width:min(960px,92vw);margin:48px auto}h1{font-size:clamp(28px,5vw,48px);margin:0}.tag{color:var(--accent);letter-spacing:.12em;text-transform:uppercase;font-size:12px;font-weight:700}.sub,.hint{color:var(--muted);max-width:660px}.hint{font-size:12px;margin:6px 0 14px}.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:16px;padding:22px;margin-top:24px;box-shadow:0 16px 50px #0005}label{display:block;color:var(--muted);margin-bottom:7px}textarea,input,select{width:100%;background:#090e1a;color:var(--text);border:1px solid var(--line);border-radius:9px;padding:12px;font:inherit}textarea{min-height:105px;resize:vertical}textarea:read-only{opacity:.7}.row{display:grid;grid-template-columns:160px 1fr auto;gap:16px;align-items:end;margin:14px 0}.resume-row{grid-template-columns:160px 1fr 180px}.keyline{display:grid;grid-template-columns:1fr auto;gap:8px}.check{display:flex;gap:8px;align-items:center;color:var(--text);margin:0 0 11px}.check input{width:auto}button{border:0;border-radius:9px;padding:12px 18px;background:var(--accent);color:#082f49;font-weight:800;cursor:pointer}.secondary{background:#27344f;color:var(--text)}button:disabled{opacity:.5;cursor:wait}.status{display:flex;justify-content:space-between;gap:12px}.pill{border:1px solid var(--line);border-radius:99px;padding:4px 10px;color:var(--muted)}.metrics{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.metric{border:1px solid var(--line);border-radius:8px;padding:5px 9px;color:var(--muted);font-size:12px}#answer{font-size:17px;white-space:pre-wrap}.event{border-left:2px solid var(--line);padding:7px 12px;margin:6px 0}.event b{color:var(--accent)}.event small{color:var(--muted)}.error{color:var(--bad)}.hidden{display:none}@media(max-width:620px){.row,.resume-row{grid-template-columns:1fr}.wrap{margin:24px auto}}
</style></head><body><main class="wrap"><div class="tag">Evidence-backed · checkpointed · observable</div><h1>Resilient Research Agent</h1><p class="sub">Give the agent a goal. It discovers sources, verifies evidence, adapts to errors, and shows every operational decision.</p>
<section class="card"><form id="form"><div class="row resume-row"><div><label for="runMode">Run mode</label><select id="runMode"><option value="new">Start new</option><option value="resume">Resume saved</option></select></div><div id="savedRunWrap" class="hidden"><label for="savedRun">Saved run</label><select id="savedRun"></select></div><div><label for="maxSteps">Total step budget</label><input id="maxSteps" type="number" min="1" max="100" value="12" required></div></div><p id="resumeHint" class="hint hidden">The original goal and provider are restored. Live runs require the provider key again.</p><label for="goal">Research goal</label><textarea id="goal" required>Compare checkpointing and retry logic for resilient agents.</textarea><div class="row"><div><label for="provider">Provider</label><select id="provider"><option value="openai">OpenAI</option><option value="groq">Groq</option><option value="libra">Libra / Company</option></select></div><div><label for="model">Model</label><input id="model" value="gpt-5-mini"></div><div><label class="check"><input id="demo" type="checkbox" checked> Offline demo</label></div></div><div id="keyRow" class="hidden"><label id="keyLabel" for="apiKey">OpenAI API key · used for this run only</label><div class="keyline"><input id="apiKey" type="password" autocomplete="off" spellcheck="false" placeholder="Enter key"><button id="reveal" class="secondary" type="button" aria-label="Show API key">Show</button></div><p class="hint">Never written to checkpoints, traces, browser storage, or server responses.</p></div><button id="run">Run agent</button></form><p id="formError" class="error"></p></section>
<section id="result" class="card hidden"><div class="status"><strong>Run progress</strong><span id="status" class="pill">starting</span></div><div class="metrics"><span id="steps" class="metric">Steps: 0</span><span id="providerRetries" class="metric">Provider retries: 0</span><span id="toolRetries" class="metric">Tool retries: 0</span><span id="validationFailures" class="metric">Validation failures: 0</span><span id="goalCoverage" class="metric">Goal coverage: pending</span><span id="citationCoverage" class="metric">Citation coverage: pending</span></div><h2>Answer</h2><div id="answer" class="sub">Working…</div><h2>Trace</h2><div id="events"></div></section></main>
<script>
const form=document.querySelector('#form'),button=document.querySelector('#run'),result=document.querySelector('#result'),statusEl=document.querySelector('#status'),answer=document.querySelector('#answer'),events=document.querySelector('#events'),formError=document.querySelector('#formError'),goalInput=document.querySelector('#goal'),providerInput=document.querySelector('#provider'),modelInput=document.querySelector('#model'),demoInput=document.querySelector('#demo'),keyRow=document.querySelector('#keyRow'),keyLabel=document.querySelector('#keyLabel'),apiKey=document.querySelector('#apiKey'),reveal=document.querySelector('#reveal'),runModeInput=document.querySelector('#runMode'),savedRunWrap=document.querySelector('#savedRunWrap'),savedRunInput=document.querySelector('#savedRun'),maxStepsInput=document.querySelector('#maxSteps'),resumeHint=document.querySelector('#resumeHint');
let savedRuns=[];
function refreshKeyRow(){keyRow.classList.toggle('hidden',demoInput.checked)}
function applySavedRun(){const run=savedRuns.find(item=>item.id===savedRunInput.value);if(!run)return;goalInput.value=run.goal;providerInput.value=run.provider;modelInput.value=run.model;demoInput.checked=run.mode==='demo';maxStepsInput.min=String(run.minimum_max_steps);maxStepsInput.value=String(Math.max(12,run.minimum_max_steps));keyLabel.textContent={openai:'OpenAI',groq:'Groq',libra:'Libra'}[run.provider]+' API key · used for this run only';refreshKeyRow();button.textContent='Resume agent'}
async function loadSavedRuns(){savedRunInput.innerHTML='<option value="">Loading...</option>';try{const r=await fetch('/api/runs'),data=await r.json();if(!r.ok)throw Error(data.error||'Could not load saved runs');savedRuns=data.runs||[];savedRunInput.innerHTML=savedRuns.length?savedRuns.map(run=>`<option value="${esc(run.id)}">${esc(run.goal.slice(0,70))} - ${esc(run.status)} at step ${run.steps}</option>`).join(''):'<option value="">No resumable runs</option>';applySavedRun()}catch(err){savedRunInput.innerHTML='<option value="">Could not load saved runs</option>';formError.textContent=err.message}}
function setRunMode(){const resume=runModeInput.value==='resume';savedRunWrap.classList.toggle('hidden',!resume);resumeHint.classList.toggle('hidden',!resume);goalInput.readOnly=resume;providerInput.disabled=resume;modelInput.readOnly=resume;demoInput.disabled=resume;maxStepsInput.min='1';button.textContent=resume?'Resume agent':'Run agent';if(resume)loadSavedRuns()}
demoInput.addEventListener('change',refreshKeyRow);
providerInput.addEventListener('change',()=>{const settings={openai:['gpt-5-mini','OpenAI'],groq:['openai/gpt-oss-20b','Groq'],libra:['gpt-5.6-sol','Libra']};const selected=settings[providerInput.value];modelInput.value=selected[0];keyLabel.textContent=selected[1]+' API key · used for this run only'});
runModeInput.addEventListener('change',setRunMode);savedRunInput.addEventListener('change',applySavedRun);
reveal.addEventListener('click',()=>{const showing=apiKey.type==='text';apiKey.type=showing?'password':'text';reveal.textContent=showing?'Show':'Hide';reveal.setAttribute('aria-label',(showing?'Show':'Hide')+' API key')});
form.addEventListener('submit',async e=>{e.preventDefault();button.disabled=true;formError.textContent='';events.innerHTML='';answer.textContent='Working…';result.classList.remove('hidden');try{const resume=runModeInput.value==='resume';if(resume&&!savedRunInput.value)throw Error('Choose a resumable run.');const payload={goal:goalInput.value,provider:providerInput.value,model:modelInput.value,demo:demoInput.checked,api_key:apiKey.value,resume,run_id:resume?savedRunInput.value:'',max_steps:Number(maxStepsInput.value)};const r=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});payload.api_key='';const data=await r.json();if(!r.ok)throw Error(data.error);apiKey.value='';apiKey.type='password';reveal.textContent='Show';poll(data.id)}catch(err){formError.textContent=err.message;button.disabled=false}});
async function poll(id){try{const r=await fetch('/api/runs/'+id),d=await r.json(),q=d.quality_checks||{};statusEl.textContent=d.status;answer.textContent=d.error||d.answer||'Working…';document.querySelector('#steps').textContent='Steps: '+(d.steps||0);document.querySelector('#providerRetries').textContent='Provider retries: '+(d.provider_retries||0);document.querySelector('#toolRetries').textContent='Tool retries: '+(d.tool_retries||0);document.querySelector('#validationFailures').textContent='Validation failures: '+(d.validation_failures||0);document.querySelector('#goalCoverage').textContent='Goal coverage: '+(q.goal_terms_total===undefined?'pending':q.goal_terms_covered+'/'+q.goal_terms_total);document.querySelector('#citationCoverage').textContent='Citation coverage: '+(q.substantive_sentences===undefined?'pending':q.cited_substantive_sentences+'/'+q.substantive_sentences);events.innerHTML=(d.events||[]).map(e=>`<div class="event"><b>Step ${e.step}: ${esc(e.event)}</b><br><small>${esc(JSON.stringify(e.detail))}</small></div>`).join('');if(['complete','failed','budget_exhausted'].includes(d.status)){button.disabled=false;return}setTimeout(()=>poll(id),500)}catch(err){formError.textContent=err.message;button.disabled=false}}
function esc(s){const x=document.createElement('span');x.textContent=s;return x.innerHTML}
</script></body></html>"""


if __name__ == "__main__":
    main()
