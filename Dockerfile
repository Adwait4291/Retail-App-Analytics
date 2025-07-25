FROM python:3.12-alpine

WORKDIR /app

RUN pip install requirements.txt

COPY src// app/src/

RUN mkdir -p /app/models

ENV PYTHONPATH=/app

CMD ["python","src/pipeline.py"]
