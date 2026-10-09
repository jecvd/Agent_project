"""用 LangGraph 编排 Agent：LLM ↔ 工具 的循环"""
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict
from openai import OpenAI

from app.config import settings
from app.tools import search_products, get_order
from search.retriever import retrieve

client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

# 工具定义（和 agent.py 里一样）
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
    "search_products": search_products,
    "get_order": get_order,
}


class AgentState(TypedDict):
    messages: list



def llm_node(state: AgentState):
    """LLM 节点：看当前对话，决定要不要调工具"""
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=state["messages"],  # 直接传，都是 OpenAI dict
        tools=TOOLS,
    )
    new_msg = resp.choices[0].message.model_dump()
    return {"messages": state["messages"] + [new_msg]}


def tool_node(state: AgentState):
    """工具节点：执行 LLM 要求的工具调用"""
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.get("tool_calls", []):
        func_name = tc["function"]["name"]
        func_args = json.loads(tc["function"]["arguments"])
        print(f"[Graph] 调用 {func_name}({func_args})")
        result = TOOL_MAP[func_name](**func_args)
        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(result, ensure_ascii=False)
        })
    return {"messages": state["messages"] + results}  # 拼接


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.get("tool_calls"):
        return "tools"
    return END



# 建图
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")  # 工具执行完回到 LLM，形成循环
agent = graph.compile()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    tests = [
        "7天无理由退货能退什么？",
        "帮我推荐一款手机",
        "O20260918001 这个订单到哪了？",
    ]
    for q in tests:
        print(f"\n===== {q} =====")
        result = agent.invoke({"messages": [{"role":"user","content":q}]})
        print(result["messages"][-1]["content"])

