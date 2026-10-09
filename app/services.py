from openai import OpenAI

from app.config import settings
from search.retriever import retrieve

_client = None

def get_llm():
    global _client
    if _client is None:
        _client = OpenAI(api_key = settings.llm_api_key,base_url = settings.llm_base_url)
    return _client

SYSTEM_PROMPT = """你是京东电商平台的人工智能客服助手。
回答规则：
1. 只能根据"参考资料"回答，禁止编造参考资料中没有的内容。
2. 如果参考资料不足以回答问题，明确说"知识库中没有找到相关信息"。
3. 回答结束后，列出引用来源的标题和链接。"""

def build_prompt(question:str,refs :list[dict])->str:
    context = "\n\n".join(
        f"【资料{i+1}】 《{r['title']}》 来源:{r['source_url']}\n{r['text']}"
        for i,r in enumerate(refs)
    )
    return f"参考资料：\n{context}\n\n用户问题：{question}\n\n请根据参考资料回答。"

#调用检索函数：把用户问题做 embedding，去 Qdrant 向量库做相似度检索，返回匹配到的文档列表`refs`
def qa(question:str):
    refs = retrieve(question)
    if not refs:
        return {"answer": "知识库中没有找到相关信息。", "references": []}
    prompt = build_prompt(question,refs)
    resp = get_llm().chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system","content": SYSTEM_PROMPT},
            {"role":"user","content":prompt},
        ],
    )
    return {
        "answer":resp.choices[0].message.content,
        "references":[
            {
                "title":r["title"],"source_url": r["source_url"], "score": r["score"]
            }
            for r in refs
        ]
    }

if __name__ == "__main__":
    for q in ["7天无理由退货能退什么？", "价格保护怎么申请？"]:
        print(f"\n=== 问：{q} ===")
        res = qa(q)
        print(res["answer"])
        print("引用：", res["references"])




