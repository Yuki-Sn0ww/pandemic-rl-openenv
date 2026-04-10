FROM python:3.10-slim

WORKDIR /app

# Install all dependencies including OpenEnv SDK, OpenAI, and Pydantic
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    requests>=2.25.0 \
    pyyaml>=6.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.20.0 \
    openenv-core>=0.2.0 \
    openai>=1.0.0 \
    pydantic>=2.0.0

# Copy application code
COPY env/ env/
COPY server/ server/
COPY pandemic_rl/ pandemic_rl/
COPY inference.py .
COPY openenv.yaml .

# Expose server port
EXPOSE 8000

# Start the FastAPI server using the OpenEnv HTTP server wrapper
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
