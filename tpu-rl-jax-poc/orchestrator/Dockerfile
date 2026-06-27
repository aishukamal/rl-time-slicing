FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    grpcio \
    fastapi \
    "uvicorn[standard]" \
    pydantic

COPY orchestrator.py /app/

EXPOSE 9000
CMD ["python", "orchestrator.py"]
