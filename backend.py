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
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- 配置中心 ---
DB_PATH = "/app/data/chat_history.db"
KNOWLEDGE_DIR = "/app/data/knowledge/"
# 分流地址
OLLAMA_CPU_API = "http://ollama-svc:11434/api/generate"
OLLAMA_GPU_API = "http://ollama-gpu-svc:11434/api/generate"
# 模型定义
PRIMARY_MODEL = "llama3:70b"   # CPU 跑逻辑
BACKUP_MODEL = "qwen2.5:7b"    # CPU 备用
VISION_MODEL = "llava:latest"   # GPU 跑识图，这里注意用的是llava
# 图片目录（对应之前的挂载）
IMAGE_DIR = "/app/user-images/"

# --- 1. 自动化向量索引构建 (保持原有逻辑) ---
print(">>> 正在初始化语义向量引擎，请稍等...")
embed_model = SentenceTransformer('/app/model_file/')
vector_index = None
all_chunks = []

def save_history(p, r, m):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO history (prompt, response, model, time) VALUES (?, ?, ?, ?)",
                        (p, r, m, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    except: pass

def get_safe_path(filename):
    """
    处理文件名：防止中文导致空字符串，防止路径穿越
    """
    ext = os.path.splitext(filename)[1]
    if not ext:
        ext = ".jpg"
    # 使用时间戳+随机字符串，绝对不会重名，也绕过了中文过滤问题
    new_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    return os.path.join(IMAGE_DIR, new_name)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # 直接使用原名保存，确保前端 GET 请求能对上
        filename = file.filename
        target_path = os.path.join(IMAGE_DIR, filename)
        file.save(target_path)
        
        print(f">>> [SRE Log] 文件已落地 NFS: {target_path}")
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        print(f">>> [SRE Error] 上传失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """处理纯文本请求，发给 CPU 上的 70B"""
    data = request.json
    prompt = data.get("prompt")
    
    print(f">>> [SRE Log] 收到文本请求，准备分发至 CPU 70B 节点")
    # 这里写请求 OLLAMA_CPU_URL 的逻辑...
    return jsonify({"reply": "70B 正在思考中..."})

def build_vector_index():
    global vector_index, all_chunks
    if not os.path.exists(KNOWLEDGE_DIR): return
    local_chunks = []
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(KNOWLEDGE_DIR, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                step = 350
                for i in range(0, len(content), step):
                    chunk_text = content[i : i + 400]
                    local_chunks.append({"filename": filename, "text": chunk_text})
    if local_chunks:
        texts = [c['text'] for c in local_chunks]
        embeddings = embed_model.encode(texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        vector_index = index
        all_chunks = local_chunks
        print(f">>> 知识库加载成功！共 {len(all_chunks)} 个片段。")

build_vector_index()

def get_semantic_context(user_query, top_n=3):
    if vector_index is None: return ""
    query_vec = embed_model.encode([user_query])
    D, I = vector_index.search(np.array(query_vec).astype('float32'), top_n)
    retrieved_parts = []
    for idx in I[0]:
        if idx != -1 and idx < len(all_chunks):
            chunk = all_chunks[idx]
            retrieved_parts.append(f"--- 参考文献: {chunk['filename']} ---\n{chunk['text']}")
    return "\n\n".join(retrieved_parts)

# --- 2. 核心逻辑：分流与识图 ---
@app.route('/aigpt_api')
def aigpt_api():
    raw_prompt = request.args.get('prompt', '')
    user_prompt = urllib.parse.unquote(raw_prompt)
    image_name = request.args.get('image', '')

    final_response = ""
    used_model = ""

    # --- 场景 1：图片识图逻辑 (GPU) ---
    if image_name:
        used_model = VISION_MODEL
        img_path = os.path.join(IMAGE_DIR, image_name)
        print(f">>> [SRE Log] 正在检索路径: {img_path}")
        
        if not os.path.exists(img_path):
            return jsonify({"response": f"❌ 错误：后端未找到图片 {image_name}，请重新上传。"})

        try:
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            r = requests.post(OLLAMA_GPU_API, json={
                "model": VISION_MODEL,
                "prompt": f"请作为 SRE 专家识别此图并回答：{user_prompt}",
                "images": [img_base64],
                "stream": False
            }, timeout=120)
            
            final_response = r.json().get('response', '') # 赋值给统一变量
        except Exception as e:
            return jsonify({"response": f"识图链路故障: {e}"}), 500

    # --- 场景 2：纯文字逻辑 (CPU 70B + RAG) ---
    else:
        knowledge_context = get_semantic_context(user_prompt)
        final_prompt = (
            f"你是一位拥有 10 年经验的 TNS 资深 SRE。请严格参考以下内部文档回答问题。\n"
            f"【内部参考资料】:\n{knowledge_context}\n\n"
            f"【用户提问】: {user_prompt}\n"
            f"请用中文专业回答。"
        ) if knowledge_context else f"你是一位资深 SRE 专家，请用中文回答: {user_prompt}"
        
        target_api = OLLAMA_CPU_API
        for model in [PRIMARY_MODEL, BACKUP_MODEL]:
            try:
                r = requests.post(target_api, json={
                    "model": model,
                    "prompt": final_prompt,
                    "stream": False
                }, timeout=(5, 300))
                if r.status_code == 200:
                    used_model = model
                    final_response = r.json().get('response', '')
                    break
            except Exception as e:
                continue

    # --- 统一持久化结果 ---
    if not final_response:
        return jsonify({"response": "后端推理服务连接失败，请检查各节点状态"}), 500

    # --- 结果持久化与标识 ---
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO history (prompt, response, model, time) VALUES (?, ?, ?, ?)",
                        (user_prompt if not image_name else f"[图片:{image_name}] {user_prompt}", 
                         final_response, used_model, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    except: pass

    display_prompt = f"[图片:{image_name}] {user_prompt}" if image_name else user_prompt
    save_history(display_prompt, final_response, used_model)

    tag = "🚀 [GPU识图]" if image_name else f"🔥 [70B+RAG]" if used_model == PRIMARY_MODEL else "💡 [Qwen]"
    return jsonify({"response": f"{tag}\n\n{final_response}"})

# 保持原有路由不变 (index, search, init_db)
@app.route('/')
def index(): return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT prompt, response, time FROM history WHERE prompt LIKE ? OR response LIKE ? ORDER BY id DESC", (f'%{query}%', f'%{query}%'))
        results = [{"prompt": r[0], "response": r[1], "time": r[2]} for r in cursor.fetchall()]
    return jsonify(results)



def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, response TEXT, model TEXT, time TEXT)''')

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
