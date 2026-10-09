#这是**RAG 检索模块**，和你前面写的入库代码（build_index）成对使用：
#入库：文档切片 → 向量化 → 存入 Qdrant
#本文件：用户提问 → 问题向量化 → Qdrant 相似度检索 → 把查到的片段 + 元信息返回，给大模型做参考。
import atexit

from qdrant_client import QdrantClient

from app.config import settings
from ingestion.embedder import embed_texts

client = QdrantClient(path=str(settings.qdrant_path))

# def close_qdrant():
#     """程序退出时自动执行关闭"""
#     client.close()
#
# # 注册钩子：python程序正常退出时，自动调用close_qdrant
# atexit.register(close_qdrant)
def retrieve(query:str,limit:int|None = None)->list[dict]:
    k = limit or settings.top_k
    query_vec = embed_texts([query])[0]

    res = client.query_points(
        collection_name=settings.collection_name,
        query = query_vec,
        limit = k *2,

    )
    seen_docs = set()
    hits = res.points
    result_list = []
    for hit in hits:
        hit_dict = hit.model_dump()
        doc_id = hit_dict["payload"]["doc_id"]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        result_list.append(
        {
            "text": hit_dict["payload"]["text"],
            "title": hit_dict["payload"]["title"],
            "doc_id": hit_dict["payload"]["doc_id"],
            "tag": hit_dict["payload"]["tag"],
            "source_url": hit_dict["payload"]["source_url"],
            "score": round(hit_dict["score"], 4),
        })
        if len(result_list) >= k:
            break
    return result_list


if __name__ == "__main__":
    questions = [
        "7天无理由退货的规则是什么？",
        "退款多久能到账？",
        "什么是价格保护？",
    ]
    for q in questions:
        print(f"\n=== 问：{q} ===")
        for r in retrieve(q):
            print(f"[{r['score']}] {r['title']} | {r['doc_id']} | {r['text'][:40]}...")
