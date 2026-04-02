from kubernetes import client, config

def get_pod_diagnostic_info(pod_name: str, namespace: str = "default"):
    """
    Skill: 获取 Pod 的底层状态和事件日志，用于诊断启动失败问题。
    """
    config.load_kube_config() # 自动加载你 D 盘 k3s 的配置
    v1 = client.CoreV1Api()
    
    try:
        # 1. 获取 Pod 详细状态
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        container_status = pod.status.container_statuses[0]
        
        # 2. 获取相关 Events (这是发现 ImagePullBackOff 的关键)
        events = v1.list_namespaced_event(namespace, field_selector=f"involvedObject.name={pod_name}")
        event_msgs = [f"[{e.type}] {e.reason}: {e.message}" for e in events.items][-5:]
        
        return {
            "status": container_status.state,
            "last_events": event_msgs
        }
    except Exception as e:
        return f"Error connecting to K3s: {str(e)}"