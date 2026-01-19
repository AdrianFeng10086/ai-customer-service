"""
改进的两阶段检索策略实现
功能：先识别相关竞赛，再从该竞赛中检索相关片段，提供更连贯的回答
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import os
import time
from typing import List, Dict, Any, Optional

# API配置
API_CONFIG = {
    "huggingface_api_key": "x",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}

class TwoStageCompetitionRetriever(BaseRetriever):
    """
    两阶段竞赛检索器，先识别相关竞赛，再检索竞赛内的相关片段
    """
    
    def __init__(
        self, 
        vectordb: Chroma,
        embeddings: Any,
        competition_k: int = 1,  # 第一阶段检索竞赛数量
        chunk_k: int = 3,  # 第二阶段检索片段数量
        include_competition_overview: bool = True,  # 是否包含竞赛概览
        similarity_threshold: float = 0.5  # 相似度阈值
    ):
        """
        初始化两阶段检索器
        
        参数:
            vectordb: Chroma向量数据库
            embeddings: 嵌入模型
            competition_k: 第一阶段检索竞赛数量
            chunk_k: 第二阶段检索片段数量
            include_competition_overview: 是否包含竞赛概览
            similarity_threshold: 相似度阈值，低于此值的竞赛不会被检索
        """
        # 初始化父类
        super().__init__()
        
        # 存储参数
        self._vectordb = vectordb
        self._embeddings = embeddings
        self._competition_k = competition_k
        self._chunk_k = chunk_k
        self._include_competition_overview = include_competition_overview
        self._similarity_threshold = similarity_threshold
        self._competition_cache = {}  # 缓存竞赛数据
        self._competition_overview_cache = {}  # 缓存竞赛概览
        self._structured_info_cache = {}  # 缓存结构化信息
        
        # 初始化时分析并缓存竞赛信息
        self._analyze_competitions()
    
    def _analyze_competitions(self):
        """分析并整理数据库中的竞赛信息"""
        print(f"正在分析竞赛信息...")
        
        # 从数据库获取全部数据
        all_data = self._vectordb.get()
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]
        
        # 分析竞赛源
        competitions = {}
        structured_info = {}
        
        for i, metadata in enumerate(metadatas):
            source = metadata.get("source", "未知")
            
            # 处理结构化信息
            if metadata.get("type") == "structured_info":
                comp_name = metadata.get("competition_name", "未知竞赛")
                structured_info[source] = {
                    "content": documents[i],
                    "metadata": metadata
                }
                continue
            
            # 普通竞赛文档
            if source not in competitions:
                competitions[source] = []
            
            competitions[source].append({
                "content": documents[i],
                "metadata": metadata,
                "index": i
            })
        
        # 创建竞赛索引映射
        self._competition_cache = competitions
        self._structured_info_cache = structured_info
        
        # 为每个竞赛创建概览
        for comp_id, docs in competitions.items():
            # 排序文档块（按照chunk_id）
            sorted_docs = sorted(docs, key=lambda x: x["metadata"].get("chunk_id", 0))
            
            # 提取第一个块的内容作为概览
            if sorted_docs:
                self._competition_overview_cache[comp_id] = {
                    "content": sorted_docs[0]["content"],
                    "metadata": sorted_docs[0]["metadata"]
                }
        
        print(f"竞赛分析完成，共发现 {len(competitions)} 个竞赛")
        for comp_id in competitions:
            print(f"- {comp_id}: {len(competitions[comp_id])} 个文档块")
    
    def _get_relevant_competitions(self, query: str) -> List[Dict[str, Any]]:
        """
        第一阶段：识别与查询相关的竞赛
        
        参数:
            query: 用户查询
        
        返回:
            相关竞赛列表，每个元素包含竞赛ID、相似度和来源文档
        """
        print(f"阶段1：识别相关竞赛...")
        
        # 使用竞赛名称和整体概述构建特殊文档集
        competition_docs = []
        
        # 添加结构化信息（如果有）
        for source, info in self._structured_info_cache.items():
            # 从结构化信息源中提取实际的竞赛ID
            # 解析 'structured_01_"未来校园"智能应用专项赛.txt'
            # 提取实际竞赛ID: '01_"未来校园"智能应用专项赛'
            competition_id = None
            if source.startswith('structured_'):
                parts = source.split('_', 1)
                if len(parts) > 1:
                    competition_id = parts[1].replace('.txt', '')
            
            doc = Document(
                page_content=info["content"],
                metadata={
                    "source": source,
                    "type": "structured_info",
                    "is_overview": True,
                    "actual_competition_id": competition_id  # 存储实际竞赛ID
                }
            )
            competition_docs.append(doc)
        
        # 添加每个竞赛的第一个文档块作为概览
        for comp_id, overview in self._competition_overview_cache.items():
            doc = Document(
                page_content=overview["content"],
                metadata={
                    "source": comp_id,
                    "is_overview": True,
                    "actual_competition_id": comp_id  # 存储实际竞赛ID
                }
            )
            competition_docs.append(doc)
        
        # 获取查询的嵌入向量
        query_embedding = self._embeddings.embed_query(query)
        
        # 对每个竞赛文档计算相似度
        results = []
        for doc in competition_docs:
            doc_embedding = self._embeddings.embed_documents([doc.page_content])[0]
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # 如果相似度高于阈值，添加到结果中
            if similarity > self._similarity_threshold:
                # 使用实际竞赛ID而不是源字段
                competition_id = doc.metadata.get("actual_competition_id", doc.metadata["source"])
                results.append({
                    "competition_id": competition_id,
                    "source": doc.metadata["source"],  # 保存原始source
                    "similarity": similarity,
                    "document": doc
                })
        
        # 按相似度排序
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        # 保留前K个最相关的竞赛
        top_competitions = results[:self._competition_k]
        
        print(f"找到 {len(top_competitions)} 个相关竞赛:")
        for comp in top_competitions:
            print(f"- {comp['competition_id']} (相似度: {comp['similarity']:.4f})")
        
        return top_competitions
    
    def _get_relevant_chunks(self, query: str, competition_id: str) -> List[Document]:
        """
        第二阶段：从特定竞赛中检索相关文档块
        
        参数:
            query: 用户查询
            competition_id: 竞赛ID
        
        返回:
            相关文档块列表
        """
        print(f"阶段2：从竞赛 '{competition_id}' 中检索相关片段...")
        
        # 获取该竞赛的所有文档
        if competition_id not in self._competition_cache:
            print(f"警告: 未找到竞赛 '{competition_id}' 的文档")
            return []
        
        competition_chunks = self._competition_cache[competition_id]
        
        # 使用向量数据库的相似度搜索
        docs_with_scores = self._vectordb.similarity_search_with_score(
            query=query,
            k=self._chunk_k,
            filter={"source": competition_id}
        )
        
        # 提取文档
        result_docs = [doc for doc, score in docs_with_scores]
        
        # 如果需要包含概览，添加到结果中
        if self._include_competition_overview and competition_id in self._competition_overview_cache:
            overview = self._competition_overview_cache[competition_id]
            overview_doc = Document(
                page_content=overview["content"],
                metadata={
                    **overview["metadata"],
                    "is_overview": True
                }
            )
            
            # 将概览放在结果列表的最前面
            result_docs = [overview_doc] + result_docs
        
        print(f"检索到 {len(result_docs)} 个相关文档块")
        return result_docs
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """计算两个向量的余弦相似度"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        执行两阶段检索，获取相关文档
        
        参数:
            query: 用户查询
            run_manager: 回调管理器
        
        返回:
            相关文档列表
        """
        start_time = time.time()
        
        # 阶段1：识别相关竞赛
        relevant_competitions = self._get_relevant_competitions(query)
        
        # 如果没有找到相关竞赛，返回空列表
        if not relevant_competitions:
            print("未找到相关竞赛，执行常规全库检索")
            # 退化为常规检索
            docs = self._vectordb.similarity_search(query, k=self._chunk_k)
            elapsed = time.time() - start_time
            print(f"检索完成，用时 {elapsed:.2f} 秒")
            return docs
        
        # 阶段2：检索相关竞赛中的片段
        result_docs = []
        for comp in relevant_competitions:
            competition_id = comp["competition_id"]
            
            # 添加结构化信息（如果有）
            structured_key = f"structured_{competition_id}.txt"
            if structured_key in self._structured_info_cache:
                info = self._structured_info_cache[structured_key]
                structured_doc = Document(
                    page_content=info["content"],
                    metadata=info["metadata"]
                )
                result_docs.append(structured_doc)
                print(f"添加结构化信息: {structured_key}")
            
            # 检索竞赛文档片段
            docs = self._get_relevant_chunks(query, competition_id)
            result_docs.extend(docs)
        
        elapsed = time.time() - start_time
        print(f"两阶段检索完成，用时 {elapsed:.2f} 秒，共返回 {len(result_docs)} 个文档")
        
        # 如果没有找到任何文档，回退到常规检索
        if not result_docs:
            print("未找到任何文档，执行常规全库检索")
            docs = self._vectordb.similarity_search(query, k=self._chunk_k)
            return docs
        
        return result_docs

def create_two_stage_retriever(db_dir=None):
    """
    创建两阶段检索器实例
    
    参数:
        db_dir: 知识库目录，默认为当前目录下的knowledge_base
    
    返回:
        初始化好的两阶段检索器
    """
    if db_dir is None:
        db_dir = os.path.join(os.getcwd(), "knowledge_base")
    
    # 初始化嵌入模型
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=API_CONFIG["huggingface_api_key"],
        model_name=API_CONFIG["embedding_model"]
    )
    
    # 加载向量数据库
    vectordb = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name="competition_knowledge"
    )
    
    # 创建并返回两阶段检索器
    return TwoStageCompetitionRetriever(
        vectordb=vectordb,
        embeddings=embeddings,
        competition_k=1,  # 检索最相关的1个竞赛
        chunk_k=6,  # 从每个竞赛中检索6个片段，增加参考文档数量
        include_competition_overview=True  # 包含竞赛概览
    )

if __name__ == "__main__":
    # 测试两阶段检索器
    retriever = create_two_stage_retriever()
    
    # 示例查询
    test_query = "未来校园智能应用专项赛的报名时间是什么时候？"
    
    # 执行检索
    results = retriever.get_relevant_documents(test_query)
    
    # 打印结果
    print("\n检索结果:")
    for i, doc in enumerate(results):
        print(f"\n文档 {i+1}:")
        if doc.metadata.get("is_overview", False):
            print("(竞赛概览)")
        print(f"元数据: {doc.metadata}")
        content = doc.page_content
        print(f"内容: {content[:200]}..." if len(content) > 200 else f"内容: {content}") 