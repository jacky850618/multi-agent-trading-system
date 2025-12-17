# 实现智能体的长期记忆机制（学习能力）
# 使用 ChromaDB 作为向量数据库，结合 OpenAI 嵌入模型

import chromadb
from openai import OpenAI
from .config_sys import CONFIG_SYS
# 将过去的交易情境 + 经验教训（reflection）存储为向量。
# 在类似情境下检索历史经验，供智能体（如多空分析师、风控经理）参考，避免重复错误。
# 每个关键智能体（如 Bull、Bear、Trader、Risk Manager）都会有自己的记忆实例

class FinancialSituationMemory:
    def __init__(self, name):
        # 使用 OpenAI 的小型嵌入模型对文本进行向量化
        self.embedding_model = "text-embedding-3-small"
        # 初始化 OpenAI 客户端（指向您配置的后端）
        self.client = OpenAI(base_url=CONFIG_SYS["backend_url"])
        # 创建一个 ChromaDB 客户端（允许重置以进行测试）
        self.chroma_client = chromadb.Client(chromadb.config.Settings(allow_reset=True))
        # 创建一个集合（类似于表格）来存储情境和建议
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        # 为给定的文本生成嵌入（向量）
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        # 将新的情境和建议添加到内存中
        if not situations_and_advice:
            return

        # 偏移量确保 ID 唯一（以防以后添加新数据）
        offset = self.situation_collection.count()
        ids = [str(offset + i) for i, _ in enumerate(situations_and_advice)]

        # 分离情境及其对应的建议
        situations = [s for s, r in situations_and_advice]
        recommendations = [r for s, r in situations_and_advice]

        # 为所有情境生成嵌入
        embeddings = [self.get_embedding(s) for s in situations]

        # 将所有内容存储在 Chroma（向量数据库）中
        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in recommendations],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        # 检索与给定查询最相似的过去情境
        if self.situation_collection.count() == 0:
            return []

        # 嵌入新的/当前情境
        query_embedding = self.get_embedding(current_situation)

        # 查询集合以获取相似的嵌入
        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_matches, self.situation_collection.count()),
            include=["metadatas"],
        )

        # 返回从匹配结果中提取的推荐
        return [{'recommendation': meta['recommendation']} for meta in results['metadatas'][0]]
