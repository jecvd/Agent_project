import csv
from pathlib import Path

from app.config import settings

MAX_CHUNK_CHARS = 400

#将文件内容按字典格式,第一行作为key存储，后面每行作为value
def read_docs(csv_path:Path)->list[dict]:
    with open(csv_path,"r",encoding = "utf-8-sig") as f:
        return list(csv.DictReader(f))

#将文本内容去除/r避免污染字段，按/n分割成列表，去掉空格和回车
def split_by_paragraph(text):
    return [p.strip() for p in text.replace("\r","").split("\n") if p.strip()]

def split_long(text:str,limit:int = MAX_CHUNK_CHARS)->list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    chunks,cur = [],""
    for seg in text.split("。"):
        piece = (seg+"。").strip()
        if len(cur)+ len(piece) <= limit:
            cur += piece
        else:
            if cur:
                chunks.append(cur)
            cur = piece
    if cur:
        chunks.append(cur)

    return chunks

def chunk_docs(docs:list[dict])->list[dict]:
    chunks = []
    for doc in docs:
        for para in split_by_paragraph(doc["content"]):
            for i,piece in enumerate(split_long(para)):
                chunks.append({
                    "text":piece,
                    "doc_id":doc["doc_id"],
                    "title": doc["title"],
                    "tag": doc["tag"],
                    "source_url": doc["source_url"],
                    "chunk_index": i,
                }
                )
    return chunks

if __name__ == '__main__':
    docs = read_docs(settings.knowledge_csv)
    chunks = chunk_docs(docs)
    print(f"生成{len(docs)}文档，生成{len(chunks)}切片")
    for c in chunks[:6]:
        print("-", c["doc_id"], c["chunk_index"], len(c["text"]), "字 |", c["text"][:25])
