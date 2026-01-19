import os
import sys
import time
import re
import argparse
import pandas as pd
import PyPDF2

from typing import List, Dict, Any, Optional

# 导入自定义的 EnhancedQASystem
from enhanced_qa import EnhancedQASystem

# 问题文件路径，默认为 questions/附件2.pdf
PDF_PATH_DEFAULT = os.path.join(os.getcwd(), "questions/附件2.pdf")
# 结果输出到 result/result_2.xlsx 文件
RESULT_PATH_DEFAULT = os.path.join(os.getcwd(), "result/result_2.xlsx")


class QuestionProcessor:
    """问题处理器，负责读取 PDF 问题并生成答案"""

    def __init__(self, verbose=True, db_dir: Optional[str] = None):
        """
        参数:
            verbose: 是否输出详细日志
            db_dir: 用于 EnhancedQASystem 的知识库目录。如果为 None，则使用 EnhancedQASystem 默认的目录。
        """
        self.verbose = verbose
        # 如果没有传入 db_dir，则用默认的空目录，确保不会扫描到无关的竞赛信息
        if db_dir is None:
            # 创建一个空的 db_dir，用于存储空的数据库文件
            db_dir = os.path.join(os.getcwd(), "empty_db")
            os.makedirs(db_dir, exist_ok=True)
        # 初始化问答系统，并传入自定义的 db_dir
        self.qa_system = EnhancedQASystem(db_dir=db_dir)

    def load_questions_from_pdf(self, pdf_path: str) -> pd.DataFrame:
        """
        从 PDF 文件中提取问题，返回一个 DataFrame
        每行数据格式：问题编号 | 问题

        改进说明：
        - 使用正则表达式结合多行和 DOTALL 模式提取每个以数字开头的完整问题
        - 去掉可能的页眉“序号 问题”，并保留问题内原有的换行（仅合并多余的空行）
        """
        all_text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"
        except Exception as e:
            print(f"加载 PDF 出错: {e}")
            return pd.DataFrame()

        # 去除可能的页眉"序号 问题"
        all_text = re.sub(r'(?m)^序号\s*问题\s*$', '', all_text)

        # 使用正则表达式提取每个问题

        pattern = r'(?m)(?s)^(\d+)\s*(.*?)(?=^\d+\s|$)'
        matches = re.findall(pattern, all_text)

        data = []
        for num, question in matches:
            # 对提取的内容进行清理：
            # - 合并连续换行符为一个换行符以保留排版效果
            # - 去除首尾多余空白
            cleaned_question = re.sub(r'\n+', '\n', question).strip()
            data.append({"问题编号": f"Q{int(num):03d}", "问题": cleaned_question})

        return pd.DataFrame(data)

    def extract_key_points(self, answer: str, context_sources: List[Dict[str, Any]]) -> List[str]:
        """
        从答案中提取关键点，增强了对常见日期格式、引号标注、标签信息和关键词的识别
        """
        key_points = []

        # 1) 提取日期信息，增加更多格式，如 2024/03/15、2024-03-15、2024年03月15日 等
        date_patterns = [
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}\s*[至到-]\s*\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
            r'(报名时间[:：]\s*([\d\-年月日/]+))'
        ]
        for pattern in date_patterns:
            matches = re.finditer(pattern, answer)
            for match in matches:
                candidate = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
                key_points.append(candidate.strip())

        # 2) 提取网址
        urls = re.findall(r'(https?://[^\s]+)', answer)
        key_points.extend(urls)

        # 3) 提取组织或机构名称（匹配2到30个汉字后跟特定词）
        orgs = re.findall(r'([\u4e00-\u9fa5]{2,30}(?:大学|学院|协会|中心|委员会|学会|部|厅|局|单位))', answer)
        key_points.extend(orgs)

        # 4) 提取数字统计信息（如“共有 3 个”格式）
        stats = re.findall(r'共有\s*(\d+)\s*(?:个|项)', answer)
        for stat in stats:
            key_points.append(f"共有{stat}个")

        # 5) 扩展标签匹配，增加额外关键词
        labels = ["报名时间", "官网", "组织单位", "基本要求", "参赛作品", "字数要求", "涉及", "任务要求", "注意"]
        for label in labels:
            pattern = label + r'[:：]\s*([^，；。\n]+)'
            matches = re.findall(pattern, answer)
            for m in matches:
                key_points.append(m.strip())

        # 6) 提取引号内的重要信息，例如竞赛名称或任务名称
        quotes = re.findall(r'“([^”]+)”', answer)
        key_points.extend(quotes)

        # 如果提取的关键点太少且提供了上下文来源，则尝试从结构化信息中补充
        if len(key_points) < 2 and context_sources:
            for source in context_sources:
                if source.get("is_structured", False):
                    content = source.get("content", "")
                    for field in labels:
                        match = re.search(f"{field}[:：]\\s*([^\\n]+)", content)
                        if match and match.group(1).strip() != "未提供":
                            key_points.append(match.group(1).strip())

        # 7) 若仍然不足，利用jieba关键词提取进一步补充
        if len(key_points) < 2:
            try:
                import jieba.analyse
                # 提取前 5 个关键词
                keywords = jieba.analyse.extract_tags(answer, topK=5)
                key_points.extend(keywords)
            except ImportError:
                # 如果没有安装jieba，则跳过
                pass

        # 去重与过滤无效内容
        filtered_points = []
        for point in key_points:
            p = point.strip()
            if p and p != "未提供" and p not in filtered_points:
                filtered_points.append(p)

        # 如果仍未提取到关键点，而答案中含有“无法”、“没有”或“未提供”，则返回提示信息
        if len(filtered_points) == 0 and ("无法" in answer or "没有" in answer or "未提供" in answer):
            filtered_points = ["无相关信息"]

        return filtered_points

    def process_questions(self, questions_df: pd.DataFrame, output_path: str) -> bool:
        """处理 PDF 问题并生成答案，输出至 Excel"""
        if questions_df is None or len(questions_df) == 0:
            print("未提取到任何问题，无法处理。")
            return False

        results = []
        total_questions = len(questions_df)
        start_time = time.time()

        for i, row in questions_df.iterrows():
            question_id = row["问题编号"]
            question_text = row["问题"]
            progress = (i + 1) / total_questions * 100
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (total_questions - i - 1) if i > 0 else 0

            if self.verbose:
                print(f"[{i + 1}/{total_questions}] ({progress:.1f}%) 正在处理: {question_id}")
                print(f"问题: {question_text}")
                print(f"已用时间: {elapsed:.1f}秒, 预计剩余: {eta:.1f}秒")

            try:
                # 调用问答系统
                qa_result = self.qa_system.answer_question(question_text)
                answer = qa_result["answer"]
                sources = qa_result["sources"]

                # 提取关键点
                key_points = self.extract_key_points(answer, sources)
                key_points_text = "；".join(key_points) if key_points else "无明确关键点"

                if self.verbose:
                    print(f"关键点: {key_points_text}")
                    print(f"回答: {answer[:100]}..." if len(answer) > 100 else f"回答: {answer}")
                    print("-" * 50)

                results.append({
                    "问题编号": question_id,
                    "问题": question_text,
                    "关键点": key_points_text,
                    "回答": answer
                })

            except Exception as e:
                print(f"处理问题 {question_id} 时出错: {str(e)}")
                results.append({
                    "问题编号": question_id,
                    "问题": question_text,
                    "关键点": "处理出错",
                    "回答": f"处理时发生错误: {str(e)}"
                })

        total_elapsed = time.time() - start_time
        avg_time = total_elapsed / total_questions if total_questions > 0 else 0

        if self.verbose:
            print(f"处理完成! 共处理 {total_questions} 个问题")
            print(f"总用时: {total_elapsed:.1f}秒, 平均每题: {avg_time:.1f}秒")

        # 保存结果到 Excel
        result_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            result_df.to_excel(output_path, index=False)
            if self.verbose:
                print(f"结果已写入: {output_path}")
            return True
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
            return False


def main():
    """主函数：从 PDF 读取问题、调用问答系统并保存结果到 Excel"""
    parser = argparse.ArgumentParser(description="从 PDF 批量读取问题并输出答案到 Excel")
    parser.add_argument("--pdf", default=PDF_PATH_DEFAULT, help="输入 PDF 文件的路径")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不显示详细日志")
    args = parser.parse_args()

    pdf_path = args.pdf
    verbose = not args.quiet

    if not os.path.isfile(pdf_path):
        print(f"错误：PDF 文件不存在：{pdf_path}")
        sys.exit(1)

    # 确保结果文件的文件夹存在
    os.makedirs(os.path.dirname(RESULT_PATH_DEFAULT), exist_ok=True)
    output_file = RESULT_PATH_DEFAULT

    # 实例化处理器。如果不想扫描竞赛信息，此处传入空目录或其他你自定义的目录
    processor = QuestionProcessor(verbose=verbose, db_dir="")  # 或传入 "empty_db" 目录

    # 读取 PDF 问题
    questions_df = processor.load_questions_from_pdf(pdf_path)
    if len(questions_df) == 0:
        print("未能从 PDF 中读取到任何有效问题。")
        sys.exit(1)

    # 开始处理
    success = processor.process_questions(questions_df, output_file)
    if success:
        print(f"处理完成，结果保存在：{output_file}")
    else:
        print("处理失败")


if __name__ == "__main__":
    main()