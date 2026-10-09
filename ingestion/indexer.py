#CSV 知识文档 → 切片 → 向量化 → 建库 → 写入 → 验证
#把 CSV 里的知识文档，切成小块 → 每块转成向量 → 连同原文一起存进 Qdrant 向量数据库，之后问答时就能按语义相似度检索

import uuid  #生成唯一ID
from qdrant_client import QdrantClient   #向量数据库的客户端
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from app.config import settings
from ingestion.chunker import read_docs, chunk_docs
from ingestion.embedder import embed_texts

VECTOR_SIZE = 1024

#**本地模式**连接 Qdrant,数据存在settings.qdrant_path
client = QdrantClient(path=str(settings.qdrant_path))

def build_index():
    #读取 + 切片
    docs = read_docs(settings.knowledge_csv)
    chunks = chunk_docs(docs)
    print(f"切片完成：{len(chunks)} 块")

    #批量向量化,把text一次性全喂给模型
    vectors = embed_texts([c["text"] for c in chunks])
    print(f"向量化完成：{len(vectors)} 条，维度 {len(vectors[0])}")

    #重删除，再建表
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)
    client.create_collection(
        collection_name = settings.collection_name,
        vectors_config = VectorParams(size = VECTOR_SIZE,distance = Distance.COSINE)
    )
    points = [PointStruct(
        id = str(uuid.uuid5(uuid.NAMESPACE_URL,f"{c['doc_id']}-{c['chunk_index']}")),
        vector = vectors[i],
        payload = {"text": c["text"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "tag": c["tag"],
                "source_url": c["source_url"],
                "chunk_index": c["chunk_index"],}
                    )
        for i,c in enumerate(chunks)
        ]
    #**把向量数据写入 Qdrant 集合**的动作
    client.upsert(
        collection_name=settings.collection_name,
        points=points
    )
    count = client.count(collection_name=settings.collection_name).count
    print(f"入库完成：{count} 个向量点")
    client.close()

if __name__ == "__main__":
    build_index()
    # print(settings.knowledge_csv.resolve())  # 看程序以为文件在哪
    # print(settings.knowledge_csv.exists())  # 直接告诉你存不存在



