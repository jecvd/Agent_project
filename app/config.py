from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR/".env",
        env_file_encoding="utf-8",
        extra = "ignore"
    )

    # 百炼配置
    llm_api_key:str = ""
    llm_base_url:str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model:str = "qwen-turbo"
    embed_model: str = "text-embedding-v3"

    qdrant_path:Path = BASE_DIR/"data/qdrant"
    collection_name:str = "knowledge_docs"
    knowledge_csv:Path = BASE_DIR/"data/raw/knowledge_docs_after_sale.csv"
    top_k:int = 5

    # app/config.py 的 Settings 类里加两行：
    products_csv: Path = BASE_DIR / "data" / "mock" / "products.csv"
    orders_csv: Path = BASE_DIR / "data" / "mock" / "orders.csv"


settings = Settings()
