"""
第三部分：知识库构建工具
功能：将提取的竞赛文本转换为向量数据库，用于智能问答系统
支持：1.从文本构建向量数据库 2.导入结构化Excel数据
"""

import os                 # 操作系统功能
import sys                # 系统功能
import time               # 时间处理
import re                 # 正则表达式
import json               # JSON处理
import argparse           # 命令行参数解析
import shutil             # 用于删除文件夹

import pandas as pd       # 数据分析库，用于读取Excel

# -------------------------------
# LangChain 与 Chroma 相关导入
# -------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings  # Hugging Face接口嵌入模型
from langchain_chroma import Chroma  # LangChain 对 Chroma 的封装
from langchain_core.documents import Document  # Document类, 用于结构化数据


import chromadb
from chromadb.config import Settings

# ====== 关键：确保全局使用相同的 Chroma 配置 ======
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,

)

# API配置 - 请替换为你自己的 Hugging Face Inference API Key
API_CONFIG = {
    "huggingface_api_key": "x",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}

def setup_embedding_model():
    """
    设置和初始化嵌入模型
    """
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=API_CONFIG["huggingface_api_key"],
            model_name=API_CONFIG["embedding_model"]
        )
        print(f"  - 成功初始化嵌入模型: {API_CONFIG['embedding_model']}")
        return embeddings
    except Exception as e:
        print(f"初始化嵌入模型时出错: {str(e)}")
        raise

def process_extracted_texts(extracted_dir, chunk_size=1000, chunk_overlap=200):
    """
    处理提取的文本文件，将它们分割成适合向量化的文本块
    """
    print("  - 开始处理提取的文本文件...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_documents = []
    all_metadatas = []

    txt_files = [
        filename for filename in os.listdir(extracted_dir)
        if filename.lower().endswith('.txt')
    ]

    if not txt_files:
        print(f"警告: 在 '{extracted_dir}' 目录中未找到TXT文件!")
        return [], []

    for i, txt_filename in enumerate(sorted(txt_files), 1):
        print(f"  - 正在处理文件 [{i}/{len(txt_files)}]: {txt_filename}")
        try:
            txt_path = os.path.join(extracted_dir, txt_filename)
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 从文件名提取文档ID
            doc_id = re.sub(r'^extracted_', '', os.path.splitext(txt_filename)[0])

            chunks = text_splitter.split_text(text)
            print(f"    * 已将文本分割为 {len(chunks)} 个块")

            for j, chunk in enumerate(chunks):
                all_documents.append(chunk)
                all_metadatas.append({
                    "source": doc_id,
                    "chunk_id": j,
                    "filename": txt_filename
                })

        except Exception as e:
            print(f"    * 处理文件 {txt_filename} 时出错: {str(e)}")
            continue

    print(f"  - 文本处理完成，共生成 {len(all_documents)} 个文本块")
    return all_documents, all_metadatas

def create_vector_database(documents, metadatas, embeddings, db_dir):
    """
    创建（或重置）向量数据库
    """
    print(f"  - 开始构建向量数据库...")
    start_time = time.time()

    try:
        # 确保数据库目录干净，若已存在则先删除
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)
            print(f"  - 已清除旧数据库，将创建全新数据库")
        os.makedirs(db_dir, exist_ok=True)

        # 创建 Chroma 的客户端
        client = chromadb.PersistentClient(path=db_dir, settings=CHROMA_SETTINGS)
        collection_name = "competition_knowledge"

        # 创建集合
        collection = client.create_collection(name=collection_name)
        print(f"  - 开始向量化文本，共 {len(documents)} 个文本块")

        # 分批向量化并插入
        batch_size = 5
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))

            batch_docs = documents[start_idx:end_idx]
            batch_metadata = metadatas[start_idx:end_idx]

            ids = [f"doc_{start_idx + i}" for i in range(len(batch_docs))]
            embedded_batch = embeddings.embed_documents(batch_docs)

            collection.add(
                embeddings=embedded_batch,
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=ids
            )

        elapsed_time = time.time() - start_time
        print(f"  - 向量数据库构建完成! 用时: {elapsed_time:.2f} 秒")

        # 保存数据库信息
        db_info = {
            "document_count": len(documents),
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": API_CONFIG["embedding_model"],
            "collection_name": collection_name
        }
        with open(os.path.join(db_dir, "db_info.json"), "w", encoding="utf-8") as f:
            json.dump(db_info, f, ensure_ascii=False, indent=2)

        # 返回一个基于同一 client 的 langchain Chroma 向量库
        vectordb = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        return vectordb

    except Exception as e:
        print(f"  - 构建向量数据库时出错: {str(e)}")
        import traceback
        print(f"  - 详细错误信息:\n{traceback.format_exc()}")
        raise



def load_structured_data(excel_path):
    """
    加载Excel中的结构化竞赛数据
    """
    try:
        df = pd.read_excel(excel_path)
        structured_data = df.to_dict(orient='records')
        print(f"  - 已加载 {len(structured_data)} 条结构化竞赛信息")
        return structured_data
    except Exception as e:
        print(f"加载结构化数据时出错: {str(e)}")
        raise

def format_competition_info(competition_info):
    """
    将竞赛信息字典格式化为文本
    """
    formatted_text = f"""
竞赛信息摘要：{competition_info.get('赛项名称', '未提供')}

赛项名称: {competition_info.get('赛项名称', '未提供')}
赛道: {competition_info.get('赛道', '未提供')}
发布时间: {competition_info.get('发布时间', '未提供')}
报名时间: {competition_info.get('报名时间', '未提供')}
组织单位: {competition_info.get('组织单位', '未提供')}
官网: {competition_info.get('官网', '未提供')}

此信息摘要由系统提取，提供竞赛的核心元数据。
来源文件: {competition_info.get('源文件', '未知')}
"""
    return formatted_text.strip()

def add_to_vector_database(texts, metadatas, embeddings, db_dir):
    """
    将新文本添加到现有的向量数据库
    """
    print(f"  - 开始将结构化信息添加到向量数据库...")
    start_time = time.time()

    try:
        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"知识库目录 '{db_dir}' 不存在，无法导入数据！")

        client = chromadb.PersistentClient(path=db_dir, settings=CHROMA_SETTINGS)
        collection_name = "competition_knowledge"
        collection = client.get_collection(name=collection_name)

        # 构造 langchain Chroma 对象，用于批量添加
        vectordb = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )

        current_count = len(vectordb.get()["ids"])
        print(f"  - 当前数据库中已有 {current_count} 条记录")

        # 组装成 Document 对象
        documents = []
        for text, metadata in zip(texts, metadatas):
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        # 使用 add_documents 批量插入
        vectordb.add_documents(documents)

        new_count = len(vectordb.get()["ids"])
        added_count = new_count - current_count

        elapsed_time = time.time() - start_time
        print(f"  - 结构化信息添加完成! 用时: {elapsed_time:.2f} 秒")
        print(f"  - 成功添加 {added_count} 条记录，数据库现在共有 {new_count} 条记录")

        # 更新 db_info.json
        db_info_path = os.path.join(db_dir, "db_info.json")
        if os.path.exists(db_info_path):
            with open(db_info_path, "r", encoding="utf-8") as f:
                db_info = json.load(f)
            db_info["document_count"] = new_count
            db_info["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            db_info["has_structured_data"] = True
            with open(db_info_path, "w", encoding="utf-8") as f:
                json.dump(db_info, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"  - 添加到向量数据库时出错: {str(e)}")
        import traceback
        print(f"  - 详细错误信息:\n{traceback.format_exc()}")
        raise

def process_excel_data(excel_path, embeddings, db_dir):
    """
    处理Excel数据并添加到向量数据库的整合函数
    """
    try:
        competition_info_list = load_structured_data(excel_path)
        if not competition_info_list:
            print("没有找到可处理的竞赛信息，跳过Excel导入")
            return

        texts = []
        metadatas = []
        for info in competition_info_list:
            if info.get('赛项名称') == "处理错误":
                continue
            formatted_text = format_competition_info(info)
            texts.append(formatted_text)

            source_file = info.get('源文件', '未知')
            if source_file.startswith('extracted_'):
                source_file = source_file[10:]

            metadata = {
                "source": f"structured_{source_file}",
                "type": "structured_info",
                "competition_name": info.get('赛项名称', '未提供')
            }
            metadatas.append(metadata)

        print(f"  - 已格式化 {len(texts)} 条竞赛信息")

        add_to_vector_database(texts, metadatas, embeddings, db_dir)
        print(f"  - 成功导入Excel数据，添加了 {len(texts)} 条结构化竞赛信息到数据库")

    except Exception as e:
        print(f"处理Excel数据时出错: {str(e)}")
        raise

def main():
    """
    主程序函数 - 支持从文本构建向量数据库和从Excel导入结构化数据
    """
    parser = argparse.ArgumentParser(description="知识库构建工具 - 支持从文本构建向量数据库和导入结构化Excel数据")
    parser.add_argument("--skip-text", action="store_true", help="跳过文本处理步骤，仅导入Excel数据")
    parser.add_argument("--skip-excel", action="store_true", help="跳过Excel导入步骤，仅处理文本")
    args = parser.parse_args()

    print("="*50)
    print("竞赛知识库构建工具")
    print("="*50)

    extracted_dir = os.path.join(os.getcwd(), "extracted")  # 存放TXT文件的目录
    result_dir = os.path.join(os.getcwd(), "result")        # result目录
    excel_path = os.path.join(result_dir, "result_1.xlsx")  # Excel 文件路径
    db_dir = os.path.join(os.getcwd(), "knowledge_base")    # 知识库目录

    if not args.skip_text and not os.path.exists(extracted_dir):
        print(f"错误: 文本目录 '{extracted_dir}' 不存在!")
        sys.exit(1)

    # 如果 Excel 不存在，但没有指定跳过，就自动跳过
    if not args.skip_excel and not os.path.exists(excel_path):
        print(f"警告: Excel文件 '{excel_path}' 不存在，将跳过Excel数据导入")
        args.skip_excel = True

    try:
        print("  - 正在初始化嵌入模型...")
        embeddings = setup_embedding_model()

        # 1. 从文本构建向量数据库
        if not args.skip_text:
            documents, metadatas = process_extracted_texts(extracted_dir)
            if not documents:
                print("没有可处理的文本文件，跳过文本建库")
            else:
                vectordb = create_vector_database(documents, metadatas, embeddings, db_dir)
                print("-"*50)
                print(f"基础知识库构建完成! 数据库已保存到: {db_dir}")
                unique_sources = set([m['source'] for m in metadatas])
                print(f"共处理 {len(documents)} 个文本块，来自 {len(unique_sources)} 个文档")

        # 2. 导入Excel结构化数据
        if not args.skip_excel:
            if not os.path.exists(db_dir) and args.skip_text:
                print(f"错误: 知识库目录 '{db_dir}' 不存在，无法导入Excel数据!")
                print("请先构建基础知识库或确保目录存在")
                sys.exit(1)

            print("-"*50)
            print("开始导入结构化Excel数据...")
            process_excel_data(excel_path, embeddings, db_dir)

        print("-"*50)
        print(f"知识库构建全部完成! 数据库位置: {db_dir}")

    except Exception as e:
        print(f"构建知识库时发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
