
import os
import time
import numpy as np
from typing import Dict, List, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

from improved_retriever import create_two_stage_retriever, TwoStageCompetitionRetriever

# API配置
API_CONFIG = {
    "huggingface_api_key": "x",  # HuggingFace API密钥
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",     # 嵌入模型
    "openai_api_key": "x",         # OpenAI API密钥
    "openai_base_url": "https://api.deepseek.com",                   # API基础URL（使用DeepSeek）
    "openai_model": "deepseek-reasoner"                              # 大语言模型
}

# 自定义竞赛问答提示模板
COMPETITION_QA_TEMPLATE = """
你是一个竞赛智能客服机器人，专门回答关于竞赛的各种问题。
请基于提供的竞赛信息回答用户的问题。

竞赛信息:
{context}

用户问题: {question}

请提供一个完整、准确且有条理的回答：
1. 如果问题涉及特定的竞赛规则、日期或要求，请明确指出这些信息
2. 如果提供的信息不够完整，请指出哪些信息是可用的，哪些是缺失的
3. 如果问题不在上下文信息范围内，请诚实地表明自己不知道，不要编造信息
4. 如果上下文中有多个竞赛的信息，请确保你的回答只针对用户询问的特定竞赛

回答格式要求：
- 使用清晰的段落组织信息
- 可以使用项目符号列表增强可读性
- 保持专业且友好的语气

回答:
"""


class EnhancedQASystem:
    """
    增强型问答系统，使用两阶段检索策略和优化的提示模板
    """

    def __init__(self, db_dir=None, callbacks=None):
        """
        初始化增强型问答系统

        参数:
            db_dir: 知识库目录，允许传入空字符串来禁用扫描。
            callbacks: 回调函数列表
        """
        # 仅当 db_dir 为 None 时才使用默认的 knowledge_base 目录
        self.db_dir = db_dir if db_dir is not None else os.path.join(os.getcwd(), "knowledge_base")
        self.callbacks = callbacks if callbacks else []
        self.qa_chain = None

        # 初始化系统（添加缺失的方法实现）
        self._initialize_qa_system()

    def _initialize_qa_system(self):
        """初始化问答系统"""
        try:
            print("正在初始化增强型问答系统...")
            start_time = time.time()
            # 创建两阶段检索器
            retriever = create_two_stage_retriever(self.db_dir)
            # 创建提示模板
            prompt = PromptTemplate.from_template(COMPETITION_QA_TEMPLATE)
            # 创建大语言模型
            llm = ChatOpenAI(
                openai_api_key=API_CONFIG["openai_api_key"],
                base_url=API_CONFIG["openai_base_url"],
                model=API_CONFIG["openai_model"],
                temperature=0.3,
                streaming=True,
                verbose=True
            )
            # 创建 QA 链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",  # 使用stuff方法将所有文档合并
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
            )
            elapsed_time = time.time() - start_time
            print(f"增强型问答系统初始化完成! 用时: {elapsed_time:.2f} 秒")
        except Exception as e:
            print(f"初始化增强型问答系统时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        回答用户问题

        参数:
            question: 用户问题

        返回:
            包含答案和来源文档的字典
        """
        if not self.qa_chain:
            raise ValueError("问答系统未初始化")

        start_time = time.time()

        try:
            # 调用QA链
            result = self.qa_chain.invoke(
                {"query": question},
                {"callbacks": self.callbacks}
            )

            elapsed_time = time.time() - start_time

            # 构建来源信息
            sources = []
            for doc in result["source_documents"]:
                source_info = {
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                    "is_overview": doc.metadata.get("is_overview", False),
                    "is_structured": doc.metadata.get("type") == "structured_info"
                }
                sources.append(source_info)

            # 返回结果
            return {
                "question": question,
                "answer": result["result"],
                "sources": sources,
                "query_time": elapsed_time
            }

        except Exception as e:
            print(f"回答问题时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

            # 返回错误信息
            return {
                "question": question,
                "answer": f"很抱歉，处理您的问题时出现了错误: {str(e)}",
                "sources": [],
                "query_time": time.time() - start_time,
                "error": str(e)
            }


def test_enhanced_qa():
    """测试增强型问答系统"""

    # 创建简单的回调处理器
    class SimpleCallbackHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token, **kwargs):
            print(token, end="", flush=True)

    # 初始化系统
    qa_system = EnhancedQASystem(callbacks=[SimpleCallbackHandler()])

    # 测试问题
    test_questions = [
        "未来校园智能应用专项赛的报名时间是什么时候？",
        "3D编程模型创新设计专项赛的组织单位是谁？",
        "未来校园智能应用专项赛的官网是什么？"
    ]

    # 回答问题
    for i, question in enumerate(test_questions):
        print(f"\n\n问题 {i+1}: {question}")
        print("-" * 50)

        result = qa_system.answer_question(question)

        print("\n" + "-" * 50)
        print(f"查询耗时: {result['query_time']:.2f} 秒")
        print(f"来源数量: {len(result['sources'])}")

        # 打印来源信息
        print("\n来源信息:")
        for i, source in enumerate(result['sources']):
            print(f"\n来源 {i+1}:")
            if source["is_overview"]:
                print("(竞赛概览)")
            if source["is_structured"]:
                print("(结构化信息)")
            print(f"元数据: {source['metadata']}")
            content = source['content']
            print(f"内容: {content[:100]}..." if len(content) > 100 else f"内容: {content}")


if __name__ == "__main__":
    # 测试增强型问答系统
    test_enhanced_qa()
