import pandas as pd

from app.config import settings


def search_products(keyword: str) -> list[dict]:
    """按关键词搜索商品（模拟商品库查询）"""
    df = pd.read_csv(settings.products_csv)
    # 在商品名和规格里模糊匹配关键词
    mask = df['name'].str.contains(keyword, na=False) | df['spec'].str.contains(keyword, na=False)
    results = df[mask].head(5)
    return results[["product_id", "name", "price", "spec"]].to_dict('records')


def get_order(order_id: str) -> dict:
    """按订单号查订单状态（模拟订单系统查询）"""
    df = pd.read_csv(settings.orders_csv)
    row = df[df['order_id'] == order_id]
    if row.empty:
        return {"error": f"未找到订单 {order_id}"}
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    print("=== 搜商品：手机 ===")
    for p in search_products("手机"):
        print(p)
    print("\n=== 查订单 ===")
    orders = pd.read_csv(settings.orders_csv)
    print("订单号样例：", orders['order_id'].head(3).tolist())
    print(get_order(orders['order_id'].iloc[0]))
