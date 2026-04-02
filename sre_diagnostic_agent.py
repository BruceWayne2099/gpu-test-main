# sudo apt-get update && sudo apt-get install -y socat 这里要安装 socat 来做端口转发，如果你在宿主机上运行这个脚本。如果是小型环境用这个方法，但是如果是大型环境，那么这个开销还是很大的。
# 生产环境的话：会给 Ollama 配置一个 Service (ClusterIP) 或者 Ingress，让其他程序通过内部域名（如 ollama.default.svc.cluster.local）直接访问，这才是最正宗、开销最小的路。
# 在生产中，我们会把它也打包成一个 Deployment 跑在 K3s 内部，实现“用 AI 监控 AI”。

import time
import smtplib
from email.mime.text import MIMEText
from kubernetes import client, config
import requests
import socket

# --- 配置区 ---
GMAIL_USER = 'chenshuo955@gmail.com'
GMAIL_PASSWORD = '邮箱的应用专用密码' # 注意：去 Google 账号设置里生成
TARGET_EMAIL = 'chenshuo955@gmail.com'
OLLAMA_URL = "http://10.42.0.50:11434/api/generate"

# 用于存储已经告警过的 Pod，防止重复发送
alerted_pods = {} 

def send_email(subject, body):
    # msg = MIMEText(body)
    # msg['Subject'] = subject
    # msg['From'] = GMAIL_USER
    # msg['To'] = TARGET_EMAIL
    
    # try:
    #     with smtplib.SMTP_SSL('74.125.203.108', 465) as server: # 直接连接 Gmail 的 SMTP 服务器 IP，避免 DNS 解析问题
    #         server.login(GMAIL_USER, GMAIL_PASSWORD)
    #         server.sendmail(GMAIL_USER, TARGET_EMAIL, msg.as_string())
    #     print("📧 邮件已发送！")
    # except Exception as e:
    #     print(f"❌ 邮件发送失败: {e}")

    with open("sre_alerts.log", "a") as f:
        f.write(f"\n{'='*20}\n{subject}\n{body}\n")
    print("📝 告警已写入本地日志文件。")

def get_failed_pods():
    config.load_kube_config(config_file='/etc/rancher/k3s/k3s.yaml')
    v1 = client.CoreV1Api()
    all_pods = v1.list_pod_for_all_namespaces()
    failed_list = []
    
    for pod in all_pods.items:
        # 筛选非正常运行的 Pod
        if pod.status.phase != "Running" and pod.status.phase != "Succeeded":
            failed_list.append(pod)
    return failed_list

def diagnose_and_notify():
    config.load_kube_config(config_file='/etc/rancher/k3s/k3s.yaml')
    v1 = client.CoreV1Api()
    failed_pods = get_failed_pods()
    
    for pod in failed_pods:
        pod_id = f"{pod.metadata.namespace}/{pod.metadata.name}"
        current_status = pod.status.phase
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace

        events = v1.list_namespaced_event(namespace)
        relevant_events = [e for e in events.items if e.involved_object.name == pod_name]
        error_context = "\n".join([f"{e.reason}: {e.message}" for e in relevant_events][-5:])
        
        # 如果这个 Pod 还没告警过，或者状态变了，就开始诊断
        if alerted_pods.get(pod_id) != current_status:
            print(f"🚨 发现异常 Pod: {pod_id}，开始 AI 诊断...")
            
            # 获取 Events
            events = v1.list_namespaced_event(pod.metadata.namespace, field_selector=f"involvedObject.name={pod.metadata.name}")
            error_context = "\n".join([f"{e.reason}: {e.message}" for e in events.items][-5:])
            
            # 调用 Qwen 诊断
            prompt = f"你是一个 SRE。Pod {pod_id} 状态异常: {current_status}。事件如下:\n{error_context}\n请给出简短的根因分析和修复建议。"
            response = requests.post(OLLAMA_URL, json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False})
            diagnosis = response.json().get('response', 'AI 无法生成诊断')
            
            # 发送邮件
            send_email(f"【K3s 告警】Pod 异常: {pod_id}", f"诊断报告：\n\n{diagnosis}")
            
            # 更新状态，标记已处理
            alerted_pods[pod_id] = current_status

if __name__ == "__main__":
    print("🚀 SRE 巡检 Agent 已启动，正在守护集群...")
    while True:
        try:
            diagnose_and_notify()
        except Exception as e:
            print(f"运行时错误: {e}")
        
        time.sleep(60) # 每分钟巡检一次