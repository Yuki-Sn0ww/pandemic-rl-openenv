FROM python:3.10-slim

WORKDIR /app

# Install only core dependencies (no Gradio — headless evaluation only)
RUN pip install --no-cache-dir numpy requests pyyaml

COPY env/ env/
COPY inference.py .
COPY openenv.yaml .

# Evaluator runs: python inference.py
CMD ["python", "inference.py"]
