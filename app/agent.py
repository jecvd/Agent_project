from openai import OpenAI
import json

from app.config import settings
from app.tools import search_products, get_order
from search.retriever import retrieve

client = OpenAI(
    api_key=settings.llm_api_key,
    base_url = settings.llm_base_url,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "查询售后知识库（退货政策、价格保护、保修等规则类问题用这个）",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用户的问题原文"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "按关键词搜索商品（用户问某类商品、价格、推荐时用这个）",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "商品关键词，如：手机、耳机"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "按订单号查询订单状态（用户查自己的物流/订单时用这个）",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，如 O20260918001"}
                },
                "required": ["order_id"]
            }
        }
    }
]

TOOL_MAP = {
    "retrieve": lambda question: retrieve(question, limit=settings.top_k),
    "search_products":search_products,
    "get_order":get_order,
}
def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": "你是电商智能客服。根据用户问题自主决定调用工具，工具结果回来后用简洁中文回答。"},
        {"role": "user", "content": question},
    ]
    # 第一轮：问 LLM，看它要不要调工具
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages = messages,
        tools= TOOLS,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for ct in msg.tool_calls:
            func_name = ct.function.name
            func_args = ct.function.arguments
            print(f"[Agent] LLM 决定调用: {func_name}({func_args})")
            result = TOOL_MAP[func_name](**func_args)

            messages.append({
                "role":"tools",
                "tool_calls":ct.id,
                "content":json.dumps(result,ensure_ascii=False)
            })

            resp2 = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages
            )
            return resp2.choices[0].message.content
    return msg.content

if __name__ == "__main__":
    # 测试三个场景
    tests = [
        "7天无理由退货能退什么？",      # 应该调 retrieve
        "帮我推荐一款手机",               # 应该调 search_products
        "O20260918001 这个订单到哪了？",  # 应该调 get_order
    ]
    for q in tests:
        print(f"\n===== 用户问：{q} =====")
        print(run_agent(q))
