FROM python:3.12.3

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Deterministic scheduling: fix string-hash seed so set/dict iteration order
# (and therefore CP-SAT model construction order) is identical across processes.
# Without this, the same payload builds structurally-different models run-to-run
# → different schedules ("lúc trễ 1 lúc trễ 2").  Must be set before interpreter
# start, so it lives here (applies to both api + worker).
ENV PYTHONHASHSEED=0

ENV HOST=0.0.0.0
ENV PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]