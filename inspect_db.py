from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import json
import os

# API配置
API_CONFIG = {
    "huggingface_api_key": "x",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}

def inspect_knowledge_base():
    """检查知识库结构和内容"""
    print("="*50)
    print("知识库检查工具")
    print("="*50)
    
    try:
        # 初始化嵌入模型
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=API_CONFIG["huggingface_api_key"],
            model_name=API_CONFIG["embedding_model"]
        )
        
        # 加载向量数据库
        db_dir = os.path.join(os.getcwd(), "knowledge_base")
        if not os.path.exists(db_dir):
            print(f"错误: 知识库目录 '{db_dir}' 不存在!")
            return
        
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name="competition_knowledge"
        )
        
        # 获取数据库信息
        data = vectordb.get()
        
        # 统计信息
        doc_count = len(data["ids"])
        print(f"知识库文档总数: {doc_count}")
        
        # 分析元数据结构和来源
        sources = {}
        metadata_keys = set()
        for metadata in data["metadatas"]:
            # 收集所有元数据键
            for key in metadata:
                metadata_keys.add(key)
            
            # 统计来源
            source = metadata.get("source", "未知来源")
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        # 打印元数据结构
        print("\n元数据字段:")
        for key in sorted(metadata_keys):
            print(f"- {key}")
        
        # 打印来源统计
        print("\n来源统计:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"- {source}: {count} 个文档")
        
        # 打印样本数据
        print("\n样本数据 (前5条):")
        for i in range(min(5, doc_count)):
            print(f"\n文档 {i+1}:")
            print(f"ID: {data['ids'][i]}")
            print(f"元数据: {json.dumps(data['metadatas'][i], ensure_ascii=False)}")
            content = data['documents'][i]
            print(f"内容: {content[:200]}..." if len(content) > 200 else f"内容: {content}")
        
        # 尝试从不同来源获取样本
        print("\n不同来源样本:")
        sampled_sources = set()
        for i in range(doc_count):
            source = data['metadatas'][i].get("source", "未知来源")
            if source not in sampled_sources and len(sampled_sources) < 5:
                sampled_sources.add(source)
                print(f"\n来源 '{source}' 样本:")
                print(f"元数据: {json.dumps(data['metadatas'][i], ensure_ascii=False)}")
                content = data['documents'][i]
                print(f"内容: {content[:200]}..." if len(content) > 200 else f"内容: {content}")
                
    except Exception as e:
        print(f"检查知识库时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    inspect_knowledge_base() 