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
# 统一使用 GPU 接口，干掉 CPU 70B
OLLAMA_GPU_API = "http://ollama-gpu-svc:11434/api/generate"
VISION_MODEL = "qwen3.5:latest" 

IMAGE_DIR = "/app/user-images/"
MODEL_CACHE_PATH = "/app/model_file/"

# 建议将阈值放宽到 1.3，增加 RAG 命中率
RAG_THRESHOLD = 1.3

# --- 1. 向量引擎初始化 ---
print(">>> [SRE] 正在加载 SentenceTransformer 模型...")
try:
    # 优先从本地 PVC 加载模型
    embed_model = SentenceTransformer(MODEL_CACHE_PATH)
except Exception as e:
    print(f">>> [Critical] 本地加载失败，尝试在线模式: {e}")
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

vector_index = None
all_chunks = []

def build_vector_index():
    """扫描知识库目录并构建 FAISS 索引"""
    global vector_index, all_chunks
    if not os.path.exists(KNOWLEDGE_DIR): 
        os.makedirs(KNOWLEDGE_DIR)
    
    local_chunks = []
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            try:
                with open(os.path.join(KNOWLEDGE_DIR, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 简单的滑动窗口切片
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
        print(f">>> [Success] 知识库已刷新，共 {len(all_chunks)} 个切片。")
    return len(local_chunks)

# 启动时执行一次初始化
build_vector_index()

def get_semantic_context_with_score(query_text, top_n=2):
    """检索最相关的上下文并返回 Dist 分数"""
    if vector_index is None or not query_text:
        return "", 999.0
    
    query_vec = embed_model.encode([query_text])
    distances, indices = vector_index.search(np.array(query_vec).astype('float32'), top_n)
    min_dist = distances[0][0] if len(distances[0]) > 0 else 999.0
    
    if min_dist > RAG_THRESHOLD:
        return "", min_dist

    retrieved_parts = []
    for idx in indices[0]:
        if idx != -1 and idx < len(all_chunks):
            chunk = all_chunks[idx]
            retrieved_parts.append(f"【参考来源: {chunk['filename']}】\n{chunk['text']}")
    return "\n\n".join(retrieved_parts), min_dist

# --- 2. 数据库逻辑 ---
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
def index(): 
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)
    target_path = os.path.join(IMAGE_DIR, file.filename)
    file.save(target_path)
    return jsonify({"status": "success", "filename": file.filename})

@app.route('/aigpt_api')
def aigpt_api():
    raw_prompt = request.args.get('prompt', '')
    user_prompt = urllib.parse.unquote(raw_prompt)
    image_name = request.args.get('image', '')

    visual_keyword = ""
    img_b64 = None

    # --- 阶段 1: 视觉解析 (如果上传了图) ---
    if image_name:
        img_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 先利用 GPU 快速提取图片关键词，用于知识库检索
            try:
                v_res = requests.post(OLLAMA_GPU_API, json={
                    "model": VISION_MODEL, 
                    "prompt": "Describe this image in 5 keywords, separated by commas.",
                    "images": [img_b64], "stream": False
                }, timeout=15)
                visual_keyword = v_res.json().get('response', '').strip()
            except: pass

    # --- 阶段 2: 语义检索 ---
    # 如果有视觉关键词，优先用关键词搜知识库，否则用用户问题搜
    search_query = visual_keyword if visual_keyword else user_prompt
    knowledge_context, dist_score = get_semantic_context_with_score(search_query)

    # --- 阶段 3: Prompt 组装 ---
    system_prompt = "你是一位资深 SRE 专家，请基于专业知识回答问题。"
    if knowledge_context:
        prompt_content = f"{system_prompt}\n\n【参考手册内容】:\n{knowledge_context}\n\n【用户问题】: {user_prompt}\n请结合手册内容给出详细排查建议。"
    else:
        prompt_content = f"{system_prompt}\n\n【用户问题】: {user_prompt}"

    # --- 阶段 4: 统一请求 GPU 模型 ---
    try:
        payload = {
            "model": VISION_MODEL,
            "prompt": prompt_content,
            "stream": False
        }
        if img_b64:
            payload["images"] = [img_b64]
            
        r = requests.post(OLLAMA_GPU_API, json=payload, timeout=60)
        final_ans = r.json().get('response', 'GPU 响应超时')
    except Exception as e:
        final_ans = f"GPU 链路异常: {str(e)}"

    save_history(user_prompt, final_ans, VISION_MODEL)
    return jsonify({"response": final_ans, "debug_dist": float(dist_score)})

# --- 4. 热刷新接口 (修复 404 问题) ---
@app.route('/refresh_knowledge', methods=['POST'])
def refresh_knowledge():
    try:
        # 重新执行扫描和索引构建
        count = build_vector_index() 
        return jsonify({"status": "success", "count": count, "msg": "Knowledge base reloaded."})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)