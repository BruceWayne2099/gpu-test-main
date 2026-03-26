FROM docker.m.daocloud.io/library/python:3.9-slim

# SRE 必备环境变量：实时日志 + 减少碎片 + 中文展示
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=zh_CN.UTF-8
ENV LANGUAGE=zh_CN:zh
ENV LC_ALL=zh_CN.UTF-8
ENV HF_ENDPOINT=https://hf-mirror.com   
# 国内网不通huggingface,改用hf-mirror

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    wget \
    locales \ 
    && sed -i -e 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY backend.py .
COPY templates/ ./templates/

# 挂载卷说明
# /app/data: SQLite 数据库
# /app/model_file: 存放 Embedding 和 Reranker 模型
# /app/user-images: 用户上传的图片
VOLUME ["/app/data","/app/model_file","/app/user-images"]

EXPOSE 5000

CMD ["python", "backend.py"]