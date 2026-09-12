FROM python:3.12-slim

# Hugging Face Spaces run the container as uid 1000; /tmp is the only writable path we need.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

COPY --chown=user . /app
USER user

ENV NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "7860"]
