# python3
# Author: Scc_hy X Claude
# Func: hello-agents 0.2.8 嵌入式 Qdrant 补丁
# Run : 无独立入口。在 import hello_agents 之前 `import _qdrant_embedded` 即生效
# ===========================================================================================
"""0.2.8 官方版的 QdrantVectorStore 只有「localhost 服务端 / 云端」两种模式
（url 不设就硬连 localhost:6333），fork 0.2.9 的嵌入式本地存储没有带过来。
本模块把它补回来：

    .env 里设置 QDRANT_PATH=./memory_data/qdrant_local 后，
    QdrantVectorStore 改走 qdrant-client 内置本地模式（QdrantClient(path=...)），
    数据落盘与 fork 时代同一目录，chap08 各脚本其余部分零改动。

未设置 QDRANT_PATH 时本模块是 no-op（保持官方服务端行为，连 localhost:6333）。

两个必须一起打补丁的点：
* _initialize_client —— 换成 path 模式（collection_name/vector_size/distance 等
  属性已由 __init__ 先行赋好，这里只补 client 和建集合）；
* __del__ —— library 的析构会 close client，而本地模式下存储目录有进程级独占锁，
  同进程的多个实例（episodic/semantic/RAG pipeline、verify_isolation 探针实例）
  必须共享同一个 client，任何一个实例析构都不能把它关掉。
"""

import os

# 兜底从同目录 .env 读 QDRANT_PATH（脚本通常已提前 load_dotenv；默认不覆盖已有变量）
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

_PATH = os.getenv("QDRANT_PATH")
if _PATH:
    from qdrant_client import QdrantClient

    from hello_agents.memory.storage import qdrant_store as _qs

    _clients: dict = {}


    def _shared_client(path: str) -> QdrantClient:
        """按路径复用 client（本地模式独占目录锁，多实例必须共享一个连接）。"""
        client = _clients.get(path)
        if client is None:
            os.makedirs(path, exist_ok=True)
            client = QdrantClient(path=path)
            _clients[path] = client
        return client


    def _embedded_initialize_client(self) -> None:
        try:
            self.client = _shared_client(_PATH)
            self.client.get_collections()          # 尽早触发一次真实访问，暴露目录锁/格式问题
            self._ensure_collection()
        except Exception as e:
            raise RuntimeError(f"嵌入式 Qdrant 初始化失败（path={_PATH}）: {e}") from e


    def _noop_del(self) -> None:                   # 共享 client 交给进程退出时统一回收
        pass


    _qs.QdrantVectorStore._initialize_client = _embedded_initialize_client
    _qs.QdrantVectorStore.__del__ = _noop_del
