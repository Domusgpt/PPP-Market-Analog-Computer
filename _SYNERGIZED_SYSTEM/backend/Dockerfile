FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/ ./engine/

ENV PYTHONPATH=/app/engine
EXPOSE 8765

CMD ["python", "engine/websocket_server.py", "--host", "0.0.0.0", "--port", "8765", "--fps", "30"]
