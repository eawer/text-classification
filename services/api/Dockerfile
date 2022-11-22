FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN python -m pip install --upgrade pip
RUN python -m pip install psycopg2-binary transformers fastapi uvicorn tritonclient geventhttpclient

ADD ./scripts/main.py /app/main.py
