FROM docker.m.daocloud.io/library/python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend.py .
COPY templates/ ./templates/

VOLUME ["/app/data","/app/model_file","/app/user-images"]

EXPOSE 5000

CMD ["python", "backend.py"]