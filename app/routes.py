from fastapi import APIRouter

from app.graph import agent
from app.schemas import QARequest, QAResponse, AgentRequest
from app.services import qa

router = APIRouter()

@router.post("/qa",response_model=QAResponse)
def ask(req:QARequest):
    return qa(req.question)

# 内存里存会话历史（生产环境用 Redis，教学用 dict 就行）
sessions: dict[str, list] = {}

@router.post("/agent")
def agent_chat(req: AgentRequest):
    session_id = req.session_id
    # 取出该用户的对话历史，没有就空列表
    messages = sessions.get(session_id, [])
    # 把新问题加进去
    messages.append({"role": "user", "content": req.question})
    # 调 graph
    result = agent.invoke({"messages": messages})
    # 存回历史
    sessions[session_id] = result["messages"]
    # 返回最后一条回答
    return {"answer": result["messages"][-1]["content"]}
