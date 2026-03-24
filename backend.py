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
    """自动化构建/更新向量索引"""
    global vector_index, all_chunks
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return

    local_chunks = []
    print(f">>> [SRE] 扫描知识库路径: {KNOWLEDGE_DIR}")
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            try:
                with open(os.path.join(KNOWLEDGE_DIR, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 按照 300 字切片，保证检索粒度
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
        dimension = embeddings.shape[1]
        # 使用 L2 距离索引
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        vector_index = index
        all_chunks = local_chunks
        print(f">>> [Success] 知识库就绪！共加载 {len(all_chunks)} 个文本片段。")
    else:
        print(">>> [Warn] 知识库为空，请在 knowledge 目录下添加 .txt 文件。")

# 启动时构建索引
build_vector_index()

def get_semantic_context(query_text, top_n=2):
    """语义搜索核心：找到最相关的 txt 片段"""
    if vector_index is None or not query_text:
        return ""
    
    query_vec = embed_model.encode([query_text])
    distances, indices = vector_index.search(np.array(query_vec).astype('float32'), top_n)
    
    retrieved_parts = []
    for idx in indices[0]:
        if idx != -1 and idx < len(all_chunks):
            chunk = all_chunks[idx]
            retrieved_parts.append(f"【参考来源: {chunk['filename']}】\n内容: {chunk['text']}")
    return "\n\n".join(retrieved_parts)

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
    # --- 阶段 1: GPU 识图 (只有传了图才走这步) ---
    if image_name:
        img_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                # 强引导 Prompt：让 Qwen 只吐出关键词
                v_prompt = "Identify the person or main object in this image. Output the name only (e.g., 'Yao Ming')."
                
                r_vision = requests.post(OLLAMA_GPU_API, json={
                    "model": VISION_MODEL,
                    "prompt": v_prompt,
                    "images": [img_b64],
                    "stream": False
                }, timeout=45)
                
                visual_keyword = r_vision.json().get('response', '').strip()
                print(f">>> [Vision] 识别结果: {visual_keyword}")
            except Exception as e:
                print(f">>> [Error] 识图失败: {e}")

    # --- 阶段 2: 向量检索 (RAG) ---
    # 策略：如果有识图结果，优先用识别到的名字去检索知识库
    search_query = visual_keyword if visual_keyword else user_prompt
    knowledge_context = get_semantic_context(search_query)

    # --- 阶段 3: CPU 70B 终极推理 ---
    # 构造最终 Prompt
    final_prompt = f"""你是一位专业的资深 SRE 助手。请结合以下信息回答。
【视觉识别对象】: {visual_keyword if visual_keyword else '未提供图片'}
【知识库参考】: 
{knowledge_context if knowledge_context else '未匹配到相关文档'}

【用户提问】: {user_prompt}

请用中文给出详细回答。如果参考资料里有具体文章内容（如“像姚明一样努力学习”），请务必在回答中引用。"""

    try:
        r = requests.post(OLLAMA_CPU_API, json={
            "model": PRIMARY_MODEL,
            "prompt": final_prompt,
            "stream": False
        }, timeout=300)
        
        final_ans = r.json().get('response', 'Master 节点未返回有效响应')
    except Exception as e:
        final_ans = f"推理链路异常: {str(e)}"

    save_history(f"[Vision:{visual_keyword}] {user_prompt}", final_ans, PRIMARY_MODEL)
    return jsonify({"response": final_ans})

if __name__ == '__main__':
    init_db()
    # 容器环境下必须 listen 0.0.0.0
    app.run(host='0.0.0.0', port=5000)