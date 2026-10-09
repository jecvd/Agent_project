import sys
from pathlib import Path
from openai import OpenAI

# 把项目根目录 D:\Agent_project 添加到模块搜索路径
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings

_client = None

#单例获取客户端
def get_model():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        print("阿里云百炼Embedding客户端加载成功")
    return _client

#输入文本列表，输出向量列表
def embed_texts(texts: list[str],batch_size = 10) -> list[list[float]]:
    client = get_model()
    all_vectors = []
    for i in range(0,len(texts),batch_size):
        batch_texts = texts[i:i+batch_size]
        resp = client.embeddings.create(     #调用embedding接口，批量向量化，create是embeddings子对象的方法
            input=batch_texts,
            model=settings.embed_model
        )
        vecs = [item.embedding for item in resp.data]
        all_vectors.extend(vecs)
    return all_vectors

if __name__ == '__main__':
    vecs = embed_texts(["7天无理由退货的规则是什么", "价格保护周期是多久"])
    print(vecs)
    print(f"向量数量：{len(vecs)}，单个向量维度：{len(vecs[0])}")
