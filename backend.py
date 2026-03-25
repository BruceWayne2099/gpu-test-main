import os
import json
import uuid
import time
import datetime
import sqlite3
import urllib.parse
import numpy as np
import requests
import faiss
import base64
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
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
RAG_THRESHOLD = 1.8

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
                    step = 250
                    window_size = 400 # 增加窗口大小，包含更多上下文
                    for i in range(0, len(content), step):
                        chunk_text = content[i : i + window_size].strip()
                        if chunk_text:
                            local_chunks.append({"filename": filename, "text": chunk_text})
            except Exception as e:
                print(f">>> [Error] 读取 {filename} 失败: {e}")

    if local_chunks:
        texts = [c['text'] for c in local_chunks]
        embeddings = embed_model.encode(texts).astype('float32')
        
        # --- 核心改进：向量归一化 ---
        faiss.normalize_L2(embeddings) 
        
        # 使用 IndexFlatIP (内积索引)，在归一化后它等同于余弦相似度
        # 余弦相似度范围是 0 到 1，越接近 1 越相似
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        vector_index = index
        all_chunks = local_chunks
        print(f">>> [Success] 知识库已刷新，维度: {embeddings.shape[1]}，切片: {len(all_chunks)}")
    return len(local_chunks)

# 启动时执行一次初始化
build_vector_index()

def get_semantic_context_with_score(query_text, top_n=2):
    """检索最相关的上下文并返回 Dist 分数"""
    if vector_index is None or not query_text:
        return "", 0.0 # 初始分改为 0
    
    query_vec = embed_model.encode([query_text]).astype('float32')
    
    # --- 核心改进：查询向量也必须归一化 ---
    faiss.normalize_L2(query_vec)
    
    # search 返回的是相似度分数 (Scores)
    scores, indices = vector_index.search(query_vec, top_n)
    max_score = float(scores[0][0])
    
    # 打印到日志里，看看这次是不是 0.x 了
    print(f">>> [RAG Debug] Query: {query_text} | Similarity Score: {max_score:.4f}", flush=True)
    
    # 余弦相似度通常 0.7 以上就算很准了，我们将阈值设为 0.6
    SIM_THRESHOLD = 0.6 
    
    if max_score < SIM_THRESHOLD:
        return "", max_score

    retrieved_parts = []
    for idx in indices[0]:
        if idx != -1 and idx < len(all_chunks):
            chunk = all_chunks[idx]
            retrieved_parts.append(f"【参考来源: {chunk['filename']}】\n{chunk['text']}")
    return "\n\n".join(retrieved_parts), max_score

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
    file.save(target_path)  # 这里直接同名覆盖
    return jsonify({"status": "success", "filename": file.filename})

@app.route('/aigpt_api')
def aigpt_api():
    raw_prompt = request.args.get('prompt', '')
    user_prompt = urllib.parse.unquote(raw_prompt)
    image_name = request.args.get('image', '')

    visual_keyword = ""
    img_b64 = None

    # --- 阶段 1: 视觉解析 (如果上传了图片) ---
    if image_name:
        img_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            try:
                # 使用非流式请求快速获取图片关键词，用于辅助 RAG 检索
                v_res = requests.post(OLLAMA_GPU_API, json={
                    "model": VISION_MODEL, 
                    "prompt": "Describe this image in 5 keywords, separated by commas.",
                    "images": [img_b64], "stream": False
                }, timeout=15)
                visual_keyword = v_res.json().get('response', '').strip()
            except Exception as e:
                print(f">>> [Vision Error]: {e}")

    # --- 阶段 2: 语义检索 (RAG) ---
    # 如果有视觉关键词，优先用关键词检索，否则用用户原话
    search_query = visual_keyword if visual_keyword else user_prompt
    knowledge_context, dist_score = get_semantic_context_with_score(search_query)

    # --- 阶段 3: 强化型 Prompt 组装 ---
    # 这里加了“死命令”，强制模型必须看参考资料，尤其是针对 yaoming.txt
    system_prompt = "你是一位资深 SRE 专家助手。请严格根据提供的【参考手册】回答问题，不要使用外部过时信息。"
    
    if knowledge_context:
        prompt_content = f"""{system_prompt}

【参考手册内容（高优先级）】:
{knowledge_context}

【用户提问】: {user_prompt}

请结合手册给出诊断建议（如果是关于姚明或学习的，请务必引用手册中的核心语录）："""
    else:
        prompt_content = f"{system_prompt}\n\n【用户提问】: {user_prompt}"

    # --- 阶段 4: 构建流式生成器 ---
    def generate():
        # 重要：先发一个包含距离分数的数据包，让前端显示 Dist 标签
        dist_packet = {"debug_dist": float(dist_score)}
        yield f"data: {json.dumps(dist_packet)}\n\n"

        payload = {
            "model": VISION_MODEL,
            "prompt": prompt_content,
            "stream": True 
        }
        if img_b64:
            payload["images"] = [img_b64]

        full_response_text = ""
        try:
            # stream=True 开启流式读取 Ollama 响应
            with requests.post(OLLAMA_GPU_API, json=payload, timeout=300, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        chunk_str = line.decode('utf-8')
                        
                        # 尝试提取文本内容用于保存历史
                        try:
                            chunk_json = json.loads(chunk_str)
                            if 'response' in chunk_json:
                                full_response_text += chunk_json['response']
                        except: pass
                        
                        # 将 Ollama 的原始 chunk 实时转发给前端
                        yield f"data: {chunk_str}\n\n"
            
            # 生成结束，保存对话到 SQLite
            if full_response_text:
                save_history(user_prompt, full_response_text, VISION_MODEL)

        except Exception as e:
            # 捕获链路超时等异常并通知前端
            yield f"data: {{\"error\": \"GPU 推理链路异常: {str(e)}\"}}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

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