FROM python:3.14.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY evals ./evals
COPY tests ./tests

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/.runs \
    && chown -R agent:agent /app/.runs

USER agent

EXPOSE 8765

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=2)" || exit 1

CMD ["research-agent-ui", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
