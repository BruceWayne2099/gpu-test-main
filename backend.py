import os
import re
import uuid
import time
import datetime
import sqlite3
import urllib.parse
import numpy as np
import requests
import faiss
import base64
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- SRE 级配置中心 ---
DB_PATH = "/app/data/chat_history.db"
KNOWLEDGE_DIR = "/app/data/knowledge/"
OLLAMA_CPU_API = "http://ollama-70b-svc:11434/api/generate" 
OLLAMA_GPU_API = "http://ollama-gpu-svc:11434/api/generate"

PRIMARY_MODEL = "llama3:70b" 
VISION_MODEL = "qwen3.5:latest" 
IMAGE_DIR = "/app/user-images/"
MODEL_CACHE_PATH = "/app/model_file/" # 确保 SentenceTransformer 在这里

# L2 距离越小越匹配。通常 1.1 以下算精准，1.3 以上基本就是胡扯了。
RAG_THRESHOLD = 1.1

# --- 1. 向量引擎初始化 ---
print(">>> [SRE] 正在加载 SentenceTransformer 模型...")
# 设置 local_files_only=True 确保不连外网
try:
    embed_model = SentenceTransformer(MODEL_CACHE_PATH)
except Exception as e:
    print(f">>> [Critical] 模型加载失败，请检查 {MODEL_CACHE_PATH}: {e}")
    # 备用方案：如果本地没有，尝试自动下载（需联网）
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

vector_index = None
all_chunks = []

def build_vector_index():
    global vector_index, all_chunks
    if not os.path.exists(KNOWLEDGE_DIR): os.makedirs(KNOWLEDGE_DIR)
    
    local_chunks = []
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            try:
                with open(os.path.join(KNOWLEDGE_DIR, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    step = 300
                    for i in range(0, len(content), step):
                        chunk_text = content[i : i + 350].strip()
                        if chunk_text:
                            local_chunks.append({"filename": filename, "text": chunk_text})
            except Exception as e:
                print(f">>> [Error] 读取 {filename} 失败: {e}")

    if local_chunks:
        texts = [c['text'] for c in local_chunks]
        embeddings = embed_model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        vector_index = index
        all_chunks = local_chunks
        print(f">>> [Success] 知识库已就绪，共 {len(all_chunks)} 个切片。")

build_vector_index()

def get_semantic_context_with_score(query_text, top_n=2):
    """
    不仅返回文本，还返回最小距离评分
    """
    if vector_index is None or not query_text:
        return "", 999.0
    
    query_vec = embed_model.encode([query_text])
    distances, indices = vector_index.search(np.array(query_vec).astype('float32'), top_n)
    
    min_dist = distances[0][0] if len(distances[0]) > 0 else 999.0
    
    # 如果距离超过阈值，直接判定为“未命中”，返回空字符串
    if min_dist > RAG_THRESHOLD:
        return "", min_dist

    retrieved_parts = []
    for idx in indices[0]:
        if idx != -1 and idx < len(all_chunks):
            chunk = all_chunks[idx]
            retrieved_parts.append(f"【参考来源: {chunk['filename']}】\n{chunk['text']}")
    
    return "\n\n".join(retrieved_parts), min_dist

# --- 2. 数据库与工具函数 ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS history 
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        prompt TEXT, response TEXT, model TEXT, time TEXT)''')

def save_history(p, r, m):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO history (prompt, response, model, time) VALUES (?, ?, ?, ?)",
                        (p, r, m, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    except: pass

# --- 3. 路由定义 ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No filename"}), 400
    
    # 保持原文件名，方便前端展示
    target_path = os.path.join(IMAGE_DIR, file.filename)
    file.save(target_path)
    return jsonify({"status": "success", "filename": file.filename})

@app.route('/aigpt_api')
def aigpt_api():
    raw_prompt = request.args.get('prompt', '')
    user_prompt = urllib.parse.unquote(raw_prompt)
    image_name = request.args.get('image', '')

    visual_keyword = ""
    # --- 阶段 1: 视觉解析 ---
    if image_name:
        img_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                # 识图引导，让它只出名字/关键词
                v_prompt = "Identify the core object or person in this image. Give me the name only."
                r_vision = requests.post(OLLAMA_GPU_API, json={
                    "model": VISION_MODEL, "prompt": v_prompt,
                    "images": [img_b64], "stream": False
                }, timeout=45)
                visual_keyword = r_vision.json().get('response', '').strip()
                print(f">>> [Vision Log] 识图关键词: {visual_keyword}")
            except Exception as e:
                print(f">>> [Vision Error] {e}")

    # --- 阶段 2: 意图感知检索 (第一重保险落地) ---
    search_query = visual_keyword if visual_keyword else user_prompt
    knowledge_context, dist_score = get_semantic_context_with_score(search_query)

    # --- 阶段 3: 动态 Prompt 构建 (第二重保险落地) ---
    if knowledge_context:
        # RAG 模式：知识库命中
        print(f">>> [SRE Logic] 命中知识库 (Distance: {dist_score:.4f}) -> 启用 RAG 增强模式")
        final_prompt = (
            f"你是一位资深 SRE 专家。请结合以下参考资料回答问题。\n"
            f"【参考资料】:\n{knowledge_context}\n\n"
            f"【用户提问】: {user_prompt}\n"
            f"请确保回答中包含参考资料的关键信息。"
        )
    else:
        # 通用模式：知识库未命中
        print(f">>> [SRE Logic] 知识库未命中 (Distance: {dist_score:.4f}) -> 启用通用推理模式")
        final_prompt = (
            f"你是一位资深 SRE 专家。请基于你的专业知识回答以下问题。\n"
            f"【提问】: {user_prompt}"
        )

    # --- 阶段 4: 请求 CPU 70B ---
    try:
        r = requests.post(OLLAMA_CPU_API, json={
            "model": PRIMARY_MODEL, "prompt": final_prompt, "stream": False
        }, timeout=300)
        final_ans = r.json().get('response', '响应超时')
    except Exception as e:
        final_ans = f"链路异常: {str(e)}"

    return jsonify({"response": final_ans, "debug_dist": float(dist_score)})

if __name__ == '__main__':
    init_db()
    # 容器环境下必须 listen 0.0.0.0
    app.run(host='0.0.0.0', port=5000)