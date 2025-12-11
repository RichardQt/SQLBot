"""
推荐问题池批量生成脚本
调用 /api/v1/question_pool/generate 接口，批量生成问题并存入数据库表 recommended_question_pool
"""
import requests

# ========== 配置区 ==========
BASE_URL = "http://localhost:8000/api/v1"  # 后端地址（注意是/api/v1）
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwiYWNjb3VudCI6ImFkbWluIiwib2lkIjoxLCJleHAiOjE3NjU0NDE2ODgsImlhdCI6MTc2NTM1NTI4OCwianRpIjoiMTQ1YmE5NDktODI1ZC00ZWQ1LWFhNDktNTdmYjdkNjdmMTNkIiwiaXNzIjoiU1FMQm90In0.aXYxfCYaz4WgYamLDI3AJAL-SazVGZxTIbZ7SPBql0E"
DATASOURCE_ID = 3                        # 数据源ID（普法文章数据库）
COUNT = 100                              # 生成问题数量
# ============================

# 注意：请求头使用 X-SQLBOT-TOKEN
headers = {"X-SQLBOT-TOKEN": f"Bearer {TOKEN}"}


def generate_questions():
    """批量生成问题并存入数据库"""
    url = f"{BASE_URL}/question_pool/generate"
    payload = {"datasource_id": DATASOURCE_ID, "count": COUNT}
    
    print(f"正在为数据源 {DATASOURCE_ID} 生成 {COUNT} 个推荐问题...")
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    
    if resp.status_code == 200:
        result = resp.json()
        data = result.get('data', result)  # 兼容有无 data 包装
        print(f"✅ 生成成功！")
        print(f"   本次新增: {data.get('generated_count', 0)} 条")
        print(f"   问题池总量: {data.get('total_count', 0)} 条")
        return data
    else:
        print(f"❌ 生成失败: {resp.status_code}")
        print(f"   响应: {resp.text}")
        return None


def get_pool_count():
    """查询问题池数量"""
    url = f"{BASE_URL}/question_pool/count/{DATASOURCE_ID}"
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return data.get("count", 0)
    return 0


def get_random_questions(count=4):
    """随机获取问题"""
    url = f"{BASE_URL}/question_pool/random/{DATASOURCE_ID}"
    resp = requests.get(url, params={"count": count}, headers=headers, timeout=30)
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        return data.get("questions", [])
    return []


def list_questions(page=1, page_size=20):
    """分页查询问题列表"""
    url = f"{BASE_URL}/question_pool/list/{DATASOURCE_ID}"
    resp = requests.get(url, params={"page": page, "page_size": page_size}, headers=headers, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    return None


if __name__ == "__main__":
    # 1. 生成问题（自动存入 recommended_question_pool 表）
    generate_questions()
    
    # 2. 查看当前数量
    print(f"\n当前问题池数量: {get_pool_count()}")
    
    # 3. 随机取4个问题
    questions = get_random_questions(4)
    print(f"\n随机问题示例:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
