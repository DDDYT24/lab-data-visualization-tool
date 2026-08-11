FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/home/labviz/.cache/matplotlib \
    LABVIZ_DATABASE_PATH=/home/labviz/.labviz/labviz-v2.db \
    LABVIZ_OBJECT_STORAGE_ROOT=/home/labviz/.labviz/objects

RUN groupadd --system labviz && useradd --system --gid labviz --create-home labviz

WORKDIR /app

COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY labviz_api ./labviz_api
COPY scripts ./scripts

RUN mkdir -p /home/labviz/.cache/matplotlib /home/labviz/.labviz/objects \
    && chown -R labviz:labviz /home/labviz

USER labviz

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["python", "-m", "uvicorn", "labviz_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
