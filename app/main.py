from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.routes import router

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="电商售后知识库问答", version="1.0")
app.include_router(router)


@app.get("/")
def index():
    # 返回聊天页面（而不是之前那个 JSON 提示）
    return FileResponse(BASE_DIR / "static" / "index.html")
