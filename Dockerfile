FROM python:3.10-slim

WORKDIR /app

# Install core + server dependencies
RUN pip install --no-cache-dir numpy requests pyyaml fastapi uvicorn

COPY env/ env/
COPY server/ server/
COPY inference.py .
COPY openenv.yaml .

# Expose server port
EXPOSE 8000

# Start the FastAPI server for Phase 2 evaluation
# The evaluator sends HTTP requests to /reset, /step, /state
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
