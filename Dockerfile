FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app

COPY requirements.txt .
COPY llmcord.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "llmcord.py"]
