import os
import requests
import json
from openai import OpenAI

# ================= 配置区 =================
BASE_URL = "http://127.0.0.1:23333/v1"
MODEL_NAME = "gpt-4o-text-only"  # 你之前 curl 查到的模型 ID
API_KEY = "sk-123456"
# ==========================================

def test_diagnostics():
    print("开始诊断测试...\n")

    # 1. 检查环境变量 (Proxy)
    print("[1/4] 检查代理环境变量...")
    proxies = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    if proxies:
        print(f"   ⚠️ 发现代理设置: {proxies}")
        print("   正在尝试在当前进程中清理代理...")
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["all_proxy"] = ""
        os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    else:
        print("   ✅ 未发现系统代理设置。")

    # 2. 测试服务器连通性 (使用 requests 直接访问)
    print(f"\n[2/4] 测试服务器连通性 (GET {BASE_URL}/models)...")
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            available_models = [m['id'] for m in models['data']]
            print(f"   ✅ 连接成功！")
            print(f"   可用模型列表: {available_models}")
            if MODEL_NAME not in available_models:
                print(f"   ❌ 错误: 配置的模型 '{MODEL_NAME}' 不在可用列表中！")
        else:
            print(f"   ❌ 失败: 服务器返回状态码 {resp.status_code}")
    except Exception as e:
        print(f"   ❌ 无法连接到服务器: {e}")
        return

    # 3. 测试 API 调用 (使用 OpenAI SDK)
    print(f"\n[3/4] 测试 OpenAI SDK 调用 (POST {BASE_URL}/chat/completions)...")
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=10
        )
        content = response.choices[0].message.content
        print(f"   ✅ 调用成功！")
        print(f"   模型回复: {content}")
    except Exception as e:
        print(f"   ❌ SDK 调用失败！")
        print(f"   错误信息: {e}")
        if "403" in str(e):
            print("   💡 提示: 403 通常表示请求被代理拦截或防火墙屏蔽。")
        elif "404" in str(e):
            print("   💡 提示: 404 表示路径错误或模型名称不匹配。")
        elif "405" in str(e):
            print("   💡 提示: 405 表示方法错误，请检查 Base URL 是否多加了后缀。")

    # 4. 检查 URL 拼接
    print(f"\n[4/4] 检查路径拼接...")
    full_url = f"{BASE_URL}/chat/completions"
    print(f"   你的完整请求地址将是: {full_url}")
    print("   (如果看到 /v1/v1/chat/completions，请把 BASE_URL 里的 /v1 删掉)")

if __name__ == "__main__":
    test_diagnostics()