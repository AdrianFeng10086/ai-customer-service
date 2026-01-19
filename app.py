
import streamlit as st
import os
import time
import sys
import io
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List
import datetime
import re

# 导入增强型问答系统
from enhanced_qa import EnhancedQASystem

# 设置页面配置
st.set_page_config(
    page_title="竞赛智能客服机器人",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 捕获后台进程输出的处理器
class ProcessCaptureHandler:
    def __init__(self):
        self.captured_output = io.StringIO()
        self.original_stdout = sys.stdout
        self.capture_active = False
        self.output_lines = []
        self.timestamp_start = None
        self.detailed_logs = []
        self.retrieval_stages = {
            "competition_identification": [],
            "document_retrieval": [],
            "llm_processing": [],
            "general": []
        }

    def start_capture(self):
        """开始捕获标准输出"""
        self.captured_output = io.StringIO()
        self.capture_active = True
        sys.stdout = self.captured_output
        self.timestamp_start = time.time()
        self.detailed_logs = []
        self.retrieval_stages = {
            "competition_identification": [],
            "document_retrieval": [],
            "llm_processing": [],
            "general": []
        }
        # 记录开始时间
        self._add_log("系统处理开始", "general", is_milestone=True)

    def _add_log(self, message, category="general", is_milestone=False):
        """添加一条带时间戳的日志"""
        elapsed = time.time() - self.timestamp_start if self.timestamp_start else 0
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "elapsed": f"{elapsed:.3f}s",
            "message": message,
            "is_milestone": is_milestone
        }
        self.detailed_logs.append(log_entry)
        if category in self.retrieval_stages:
            self.retrieval_stages[category].append(log_entry)

    def end_capture(self):
        """结束捕获并返回捕获的输出"""
        if self.capture_active:
            sys.stdout = self.original_stdout
            self.capture_active = False

            # 获取捕获的输出并分割为行
            output = self.captured_output.getvalue()
            self.output_lines = output.split('\n')

            # 过滤并处理捕获的输出
            processed_lines = []
            in_stage_1 = False
            in_stage_2 = False

            for line in self.output_lines:
                # 过滤掉LLM生成的token 
                if len(line) > 1 or line.strip() == '':
                    processed_lines.append(line)

                    # 识别阶段1和阶段2
                    if "阶段1：识别相关竞赛" in line:
                        in_stage_1 = True
                        in_stage_2 = False
                        self._add_log(line, "competition_identification", is_milestone=True)
                    elif "开始执行检索链" in line or "retrieving relevant documents" in line:
                        # 检查是否已经在阶段1中，并且还没有进入阶段2
                        if not in_stage_2 and in_stage_1:
                            in_stage_2 = True
                            stage2_msg = "阶段2：从相关竞赛中检索文档"
                            processed_lines.append(stage2_msg)
                            self._add_log(stage2_msg, "document_retrieval", is_milestone=True)

                    # 根据行内容或当前阶段分类日志
                    if in_stage_1 and not in_stage_2:
                        # 阶段1相关日志
                        if any(kw in line for kw in ["相关竞赛", "相似度:", "竞赛识别"]):
                            self._add_log(line, "competition_identification")
                    elif in_stage_2:
                        # 阶段2相关日志
                        if any(kw in line for kw in ["检索链", "文档", "检索到", "相关文档块"]):
                            self._add_log(line, "document_retrieval")
                    elif "开始API请求" in line or "生成答案" in line or "API响应完成" in line:
                        self._add_log(line, "llm_processing", "API" in line)
                    else:
                        self._add_log(line, "general")

            self.output_lines = processed_lines
            # 记录结束时间
            self._add_log("系统处理完成", "general", is_milestone=True)
            return output
        return ""

    def get_retrieval_process_info(self):
        """提取检索过程相关的行，并整理为清晰的格式"""
        # 收集原始信息
        raw_info = []
        stage1_lines = []
        stage2_lines = []
        closing_lines = []

        current_stage = None

        # 第一步：按阶段分类所有行
        for line in self.output_lines:
            # 识别阶段1开始
            if "阶段1：识别相关竞赛" in line:
                current_stage = "stage1"
                stage1_lines.append(line)
            # 识别阶段2开始标记
            elif any(marker in line for marker in
                     ["阶段2：从相关竞赛中检索文档", "阶段2：从竞赛", "开始执行检索链"]) and current_stage == "stage1":
                if "阶段2：" not in line:
                    line = "阶段2：从相关竞赛中检索文档"
                current_stage = "stage2"
                # 只添加第一个阶段2标记，避免重复
                if not stage2_lines:
                    stage2_lines.append(line)
            # 识别结束标记
            elif "两阶段检索完成" in line or "检索完成，用时" in line:
                current_stage = "closing"
                closing_lines.append(line)
            # 根据当前阶段归类其他行
            elif current_stage == "stage1":
                if "找到" in line and "个相关竞赛" in line:
                    stage1_lines.append(line)
                elif "相似度:" in line:
                    stage1_lines.append(line)
                elif "添加结构化信息" in line:
                    stage1_lines.append(line)
            elif current_stage == "stage2":
                # 过滤掉重复的阶段2标记
                if not any(marker in line for marker in ["阶段2：", "开始执行检索链"]):
                    if "检索到" in line and "个相关文档块" in line:
                        # 确保只添加一次文档块数量信息
                        if not any("检索到" in l and "个相关文档块" in l for l in stage2_lines):
                            stage2_lines.append(line)
                    elif "文档" in line:
                        stage2_lines.append(line)

        # 第二步：格式化每个阶段的输出
        # 阶段1
        if stage1_lines:
            raw_info.append("**🔍 阶段1：识别相关竞赛**")
            # 格式化竞赛识别结果
            for line in stage1_lines:
                if "阶段1：识别相关竞赛" in line:
                    continue  # 跳过标题，因为我们已经添加了格式化的标题
                elif "找到" in line and "个相关竞赛" in line:
                    raw_info.append(f"**{line}**")
                elif line.strip().startswith("- ") and "相似度:" in line:
                    raw_info.append(f"* {line}")
                elif "添加结构化信息" in line:
                    raw_info.append(f"* {line}")
                else:
                    raw_info.append(line)

        # 阶段2
        if stage2_lines:
            raw_info.append("")  # 添加空行分隔
            raw_info.append("**📄 阶段2：从相关竞赛中检索文档**")
            # 格式化文档检索结果
            for line in stage2_lines:
                if "阶段2：" in line:
                    continue  # 跳过标题，因为我们已经添加了格式化的标题
                elif "检索到" in line and "个相关文档块" in line:
                    raw_info.append(f"**{line}**")
                elif "文档" in line and "来源" in line:
                    raw_info.append(f"* {line}")
                else:
                    raw_info.append(line)

        # 结束信息
        if closing_lines:
            raw_info.append("")  # 添加空行分隔
            raw_info.append("**✓ 检索完成**")
            for line in closing_lines:
                raw_info.append(f"**{line}**")

        return raw_info

    def get_detailed_process_info(self):
        """获取详细的处理过程信息，按阶段分类"""
        return {
            "详细日志": self.detailed_logs,
            "竞赛识别阶段": self.retrieval_stages["competition_identification"],
            "文档检索阶段": self.retrieval_stages["document_retrieval"],
            "LLM处理阶段": self.retrieval_stages["llm_processing"],
            "总处理时间": f"{time.time() - self.timestamp_start:.3f}秒" if self.timestamp_start else "未知"
        }

    def get_performance_metrics(self):
        """获取性能指标"""
        if not self.timestamp_start:
            return {}

        total_time = time.time() - self.timestamp_start

        # 分析各阶段时间
        stage_times = {}
        competition_id_time = 0
        document_retrieval_time = 0
        llm_processing_time = 0

        # 计算匹配的竞赛数和文档块数
        competition_count = 0
        doc_chunk_count = 0

        for line in self.output_lines:
            if "找到" in line and "个相关竞赛" in line:
                match = re.search(r'找到\s*(\d+)\s*个相关竞赛', line)
                if match:
                    competition_count = int(match.group(1))

            if "检索到" in line and "个相关文档块" in line:
                match = re.search(r'检索到\s*(\d+)\s*个相关文档块', line)
                if match:
                    doc_chunk_count = int(match.group(1))

        # 尝试根据时间戳估算各阶段时间
        comp_id_logs = self.retrieval_stages["competition_identification"]
        doc_ret_logs = self.retrieval_stages["document_retrieval"]
        llm_logs = self.retrieval_stages["llm_processing"]

        # 辅助函数，从格式为 "36.359s" 的字符串中提取浮点数
        def extract_seconds(time_str):
            if isinstance(time_str, str) and time_str.endswith('s'):
                return float(time_str[:-1])  # 移除末尾的's'再转换
            return float(time_str)  # 如果已经是数字或其他格式，尝试直接转换

        if comp_id_logs and len(comp_id_logs) > 1:
            try:
                first_time = extract_seconds(comp_id_logs[0]["elapsed"])
                last_time = extract_seconds(comp_id_logs[-1]["elapsed"])
                competition_id_time = last_time - first_time
            except (ValueError, KeyError) as e:
                print(f"计算竞赛识别时间出错: {e}")
                competition_id_time = 0

        if doc_ret_logs and len(doc_ret_logs) > 1:
            try:
                first_time = extract_seconds(doc_ret_logs[0]["elapsed"])
                last_time = extract_seconds(doc_ret_logs[-1]["elapsed"])
                document_retrieval_time = last_time - first_time
            except (ValueError, KeyError) as e:
                print(f"计算文档检索时间出错: {e}")
                document_retrieval_time = 0

        if llm_logs and len(llm_logs) > 1:
            try:
                first_time = extract_seconds(llm_logs[0]["elapsed"])
                last_time = extract_seconds(llm_logs[-1]["elapsed"])
                llm_processing_time = last_time - first_time
            except (ValueError, KeyError) as e:
                print(f"计算LLM处理时间出错: {e}")
                llm_processing_time = 0

        return {
            "总处理时间": f"{total_time:.3f}秒",
            "竞赛识别时间": f"{competition_id_time:.3f}秒",
            "文档检索时间": f"{document_retrieval_time:.3f}秒",
            "LLM处理时间": f"{llm_processing_time:.3f}秒",
            "匹配竞赛数": competition_count,
            "检索文档块数": doc_chunk_count
        }


# 自定义流式输出处理器 - 同时在Streamlit界面和终端输出
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, process_capture: ProcessCaptureHandler = None):
        self.container = container
        self.text = ""
        self.run_id = None
        self.process_capture = process_capture
        self.token_count = 0
        self.start_time = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """当LLM生成新token时被调用"""
        self.text += token
        self.token_count += 1

        # 使用markdown更新，确保立即刷新
        self.container.markdown(self.text + "▌", unsafe_allow_html=True)

        # 在终端输出token
        print(token, end="", flush=True)

        # 记录处理日志
        if self.process_capture and self.token_count % 20 == 0:  # 每20个token记录一次
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.process_capture._add_log(f"已生成 {self.token_count} 个token，用时：{elapsed:.2f}秒", "llm_processing")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """当LLM开始生成时被调用"""
        if "run_id" in kwargs:
            self.run_id = kwargs["run_id"]
        # 记录开始时间
        self.start_time = time.time()
        self.token_count = 0
        # 在终端输出API请求信息
        print("\n\n---------- 开始API请求 ----------")
        print(f"提示: {prompts[0][:100]}...")

        # 显示初始化状态
        self.container.markdown("正在生成回答中...▌", unsafe_allow_html=True)

        # 记录处理日志
        if self.process_capture:
            self.process_capture._add_log("开始LLM生成", "llm_processing", is_milestone=True)
            self.process_capture._add_log(f"提示长度：{len(prompts[0])} 字符", "llm_processing")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """当LLM完成生成时被调用"""
        # 更新最终文本（移除光标）
        self.container.markdown(self.text, unsafe_allow_html=True)

        # 在终端输出API响应完成信息
        print("\n---------- API响应完成 ----------\n")

        # 记录处理日志
        if self.process_capture:
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.process_capture._add_log(f"LLM生成完成，共 {self.token_count} 个token，用时：{elapsed:.2f}秒",
                                          "llm_processing", is_milestone=True)


# 自定义Verbose回调处理器 - 打印更详细的API交互信息
class VerboseHandler(BaseCallbackHandler):
    def __init__(self, process_capture: ProcessCaptureHandler = None):
        self.process_capture = process_capture
        self.chain_start_time = None

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """当链开始执行时被调用"""
        self.chain_start_time = time.time()
        print("\n---------- 开始执行检索链 ----------")
        print("阶段2：从相关竞赛中检索文档")  # 明确标记阶段2的开始
        if "query" in inputs:
            print(f"问题: {inputs['query']}")

        # 记录处理日志
        if self.process_capture:
            self.process_capture._add_log("阶段2：从相关竞赛中检索文档", "document_retrieval", is_milestone=True)
            if "query" in inputs:
                self.process_capture._add_log(f"查询: {inputs['query']}", "document_retrieval")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """当链执行结束时被调用"""
        print("\n---------- 检索链执行完成 ----------")
        if "result" in outputs:
            print(f"生成答案长度: {len(outputs['result'])}")
        if "source_documents" in outputs:
            doc_count = len(outputs['source_documents'])
            print(f"检索到 {doc_count} 个相关文档块（增强型检索，提供更全面的信息）")

            for i, doc in enumerate(outputs['source_documents']):
                print(f"\n文档 {i + 1} 元数据: {doc.metadata}")
                print(f"文档 {i + 1} 内容: {doc.page_content[:100]}...")

        # 记录处理日志
        if self.process_capture:
            elapsed = time.time() - self.chain_start_time if self.chain_start_time else 0
            self.process_capture._add_log(f"检索链执行完成，用时：{elapsed:.2f}秒", "document_retrieval",
                                          is_milestone=True)
            if "result" in outputs:
                self.process_capture._add_log(f"生成答案长度: {len(outputs['result'])} 字符", "document_retrieval")
            if "source_documents" in outputs:
                doc_count = len(outputs['source_documents'])
                self.process_capture._add_log(f"检索到 {doc_count} 个相关文档块（增强型检索）", "document_retrieval")
                for i, doc in enumerate(outputs['source_documents']):
                    self.process_capture._add_log(f"文档 {i + 1} 来源: {doc.metadata.get('source', '未知')}",
                                                  "document_retrieval")


# 样式设置
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; /* More modern system fonts */
        background-color: #f4f6f9; /* Slightly softer light grey */
        color: #333; /* Default text color */
    }

    /* Container to constrain width slightly on very wide screens */
    .main .block-container {
        max-width: 1400px; /* Adjust as needed */
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Main header - Refined gradient and spacing */
    .main-header {
        font-size: 2.6rem; /* Slightly smaller */
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.8rem;
        background: linear-gradient(60deg, #4e54c8, #8f94fb); /* Softer blue/purple gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 1.5rem; /* More space above */
    }

    /* Sub header - Softer color */
    .sub-header {
        font-size: 1.1rem;
        color: #667;
        text-align: center;
        margin-bottom: 2.5rem; /* More space below */
    }

    /* Chat message base style - Softer corners, more padding */
    .stChatMessage {
        border-radius: 15px; /* More rounded */
        padding: 1.1rem 1.5rem; /* Increased padding */
        margin-bottom: 1.2rem; /* Increased spacing */
        box-shadow: 0 3px 6px rgba(0,0,0,0.04); /* Softer shadow */
        border: 1px solid transparent; /* Base border */
        max-width: 80%; /* Slightly narrower max width */
    }

    /* User message styling - Softer blue, clearer distinction */
    [data-testid="stChatMessageContent"]:has(div[data-testid="stMarkdownContainer"] p) {
        background-color: #e9f5ff; /* Lighter, softer blue */
        border: 1px solid #cde4ff;
        border-left: 5px solid #5c9dff; /* Clearer blue accent */
        margin-left: auto;
        margin-right: 0;
        color: #333;
    }
    /* User icon slightly adjusted */
    [data-testid="stChatMessageContent"]:has(div[data-testid="stMarkdownContainer"] p)::before {
        content: "👤";
        margin-right: 10px;
        font-size: 1.1em;
        opacity: 0.8;
    }

    /* Bot message styling - Clean white, subtle purple */
    [data-testid="stChatMessageContent"]:not(:has(div[data-testid="stMarkdownContainer"] p)) {
        background-color: #ffffff;
        border: 1px solid #e8e8e8;
        border-left: 5px solid #8a74f9; /* Softer purple accent */
        margin-right: auto;
        margin-left: 0;
        color: #333;
    }
    /* Bot icon slightly adjusted */
    [data-testid="stChatMessageContent"]:not(:has(div[data-testid="stMarkdownContainer"] p))::before {
        content: "🤖";
        margin-right: 10px;
        font-size: 1.1em;
        opacity: 0.8;
    }

    /* Source/Details Expanders - Softer look */
    .stExpander {
        border: 1px solid #e6e8eb;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: box-shadow 0.2s ease;
    }
    .stExpander:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
    }
    .stExpander header {
        font-weight: 500; /* Slightly lighter weight */
        color: #444;
        padding: 0.9rem 1.2rem;
        border-bottom: 1px solid #e6e8eb;
        border-radius: 10px 10px 0 0; /* Match container */
        background-color: #fafbfc;
    }
    .stExpander header:hover {
        background-color: #f5f7fa;
    }

    /* Styling within expanders - More padding, softer borders */
    .details-container {
        padding: 0.8rem 1.2rem;
    }
    .source-title {
        font-weight: 600; /* Slightly bolder */
        color: #4e54c8; /* Match header gradient start */
        font-size: 0.95em;
        margin-bottom: 3px; /* Space below title */
        display: block;
    }
    .structured-info, .overview-info, .details-container > div:not(.structured-info):not(.overview-info) {
        border-left: 4px solid;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        border-radius: 6px;
        background-color: #f8f9fc;
    }
    /* Use more muted accent colors */
    .structured-info { border-color: #ffc107; background-color: #fff8e1; } /* Muted Yellow */
    .overview-info { border-color: #8a74f9; background-color: #f3f0ff;} /* Muted Purple */
    .details-container > div:not(.structured-info):not(.overview-info) { border-color: #adb5bd; background-color: #f1f3f5; } /* Muted Grey */

    /* Process display - Lighter dark theme, better font rendering */
    .process-display {
        font-family: 'SF Mono', 'Consolas', 'Menlo', monospace;
        line-height: 1.65;
        padding: 1.2rem 1.5rem;
        background-color: #2d333b; /* Lighter dark */
        color: #cdd9e5; /* Softer light text */
        border-radius: 10px;
        border: 1px solid #444c56;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin-top: 1rem;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    .process-display p { margin-bottom: 8px; }
    /* Refined colors for better readability on dark bg */
    .process-stage { color: #8ade9a; font-weight: 600; } /* Brighter Green */
    .process-result { color: #f0d88c; margin-left: 15px; } /* Brighter Yellow */
    .process-display b { color: #ffa3a3; } /* Softer Red */

    /* Detailed Process Metrics & Timeline - More spacing, refined timeline */
    .detailed-process {
        font-family: 'SF Mono', 'Consolas', 'Menlo', monospace;
        line-height: 1.6;
        padding: 1.2rem;
        background-color: #fff;
        border-radius: 10px;
        border: 1px solid #e6e8eb;
    }
    .process-summary {
        background-color: #eef2f7;
        border-left: 4px solid #6c757d; /* Neutral grey border */
        padding: 12px 18px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    .process-summary h4 { margin-bottom: 8px; color: #343a40; font-weight: 600; }

    .timeline {
        border-left: 2px solid #e1e4e8; /* Slightly darker grey */
        padding-left: 20px;
        margin-left: 8px;
    }
    .log-entry {
        margin-bottom: 10px;
        padding: 6px 10px;
        border-radius: 6px;
        background-color: #f6f8fa;
        position: relative;
        border: 1px solid #e1e4e8;
        /* Refined timeline dot */
        &:before {
            content: '';
            position: absolute;
            left: -25px; /* Adjust based on padding-left of .timeline */
            top: 10px; /* Align with text better */
            width: 10px;
            height: 10px;
            background-color: #fff;
            border: 2px solid #adb5bd; /* Grey hollow dot */
            border-radius: 50%;
        }
    }
    .log-entry.milestone {
        background-color: #e6fffa; /* Light teal */
        font-weight: 500;
        border-left: 3px solid #17c3a3; /* Teal */
        border-color: #a6f7e8;
        &:before { background-color: #17c3a3; border-color: #17c3a3; } /* Solid teal dot */
    }
    .timestamp { color: #586069; margin-right: 10px; font-size: 0.8em; }
    .elapsed { color: #4e54c8; margin-right: 12px; font-size: 0.8em; font-weight: 600; }
    .message { color: #24292e; }

    /* Metric Cards - Softer look, better alignment */
    .process-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); /* Slightly wider min */
        gap: 18px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e1e4e8;
        /* border-top: none; Remove colored top border for cleaner look */
        text-align: left; /* Align text left */
        box-shadow: 0 3px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    }
    .metric-card div:first-child { /* Label */
        font-size: 0.9em;
        color: #586069;
        margin-bottom: 8px;
        text-transform: uppercase; /* Uppercase label */
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #24292e;
        line-height: 1.2;
    }

    /* Stage specific timeline styling - Use backgrounds subtly */
    .competition-stage, .document-stage, .llm-stage {
        padding: 10px 15px;
        margin-bottom: 18px;
        border-radius: 8px;
        border: 1px solid;
    }
    .competition-stage { border-color: #ffd54f; background-color: #fff8e1;} /* Muted Orange */
    .document-stage { border-color: #90caf9; background-color: #e3f2fd;} /* Muted Blue */
    .llm-stage { border-color: #ce93d8; background-color: #f3e5f5;} /* Muted Purple */

    .competition-stage h4, .document-stage h4, .llm-stage h4 {
      margin-bottom: 12px;
      font-size: 1.05em;
      font-weight: 600;
    }
    .competition-stage h4 { color: #e65100; }
    .document-stage h4 { color: #0d47a1; }
    .llm-stage h4 { color: #4a148c; }

    /* Sidebar styling - Cleaner background */
    [data-testid="stSidebar"] {
        background-color: #fcfdff; /* Very light off-white */
        border-right: 1px solid #e1e4e8;
        padding: 2rem 1.2rem;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stHeading {
        color: #333;
        font-weight: 600;
    }

    /* Toggle switch styling */
    .stToggle label {
        color: #333;
    }

    /* Button styling - Refined */
    .stButton button {
        border-radius: 8px; /* Less rounded */
        border: 1px solid;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease, transform 0.1s ease;
    }
    /* Primary button style */
    .stButton:not(:has(button span:contains('清空聊天历史'))) button {
        border-color: #4e54c8;
        background-color: #4e54c8;
        color: white;
    }
    .stButton:not(:has(button span:contains('清空聊天历史'))) button:hover {
        background-color: #3a3f9a;
        border-color: #3a3f9a;
        transform: translateY(-1px);
    }
    /* Destructive button style (clear history) */
    .stButton:has(button span:contains('清空聊天历史')) button {
        background-color: transparent;
        border-color: #d73a49;
        color: #d73a49;
    }
    .stButton:has(button span:contains('清空聊天历史')) button:hover {
        background-color: #d73a49;
        border-color: #d73a49;
        color: white;
        transform: translateY(-1px);
    }

</style>
""", unsafe_allow_html=True)


def initialize_qa_system():
    """初始化增强型问答系统"""
    try:
        # 确保知识库目录存在
        db_dir = os.path.join(os.getcwd(), "knowledge_base")
        if not os.path.exists(db_dir):
            st.error(f"错误: 知识库目录 '{db_dir}' 不存在!")
            st.info("请先运行 `python 3_knowledge_base.py` 构建知识库")
            return None

        # 初始化系统
        qa_system = EnhancedQASystem(db_dir=db_dir, callbacks=None)

        return qa_system

    except Exception as e:
        st.error(f"初始化问答系统时出错: {str(e)}")
        print(f"初始化问答系统时出错: {str(e)}", file=sys.stderr)
        return None


def format_sources(sources):
    """格式化源文档信息为HTML，高亮结构化信息和概览"""
    html = "<div class='details-container'>"

    for i, source in enumerate(sources):
        # 确定源信息的类型和样式
        classes = []
        prefix = ""

        if source.get("is_structured", False):
            classes.append("structured-info")
            prefix = "<span class='source-title'>📋 结构化信息</span> - "
        elif source.get("is_overview", False):
            classes.append("overview-info")
            prefix = "<span class='source-title'>📑 竞赛概览</span> - "
        else:
            prefix = f"<span class='source-title'>📄 来源 {i + 1}</span> - "

        # 构建类名字符串
        class_str = " ".join(classes)
        if class_str:
            class_attr = f"class='{class_str}'"
        else:
            class_attr = ""

        # 获取元数据
        metadata = source.get("metadata", {})
        source_id = metadata.get("source", "未知来源")
        filename = metadata.get("filename", "")
        chunk_id = metadata.get("chunk_id", "")

        # 截断内容
        content = source.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."

        # 构建源信息HTML
        html += f"<div {class_attr}>"
        html += f"{prefix}"

        if filename:
            html += f"<b>文件:</b> {filename}<br>"
        else:
            html += f"<b>来源:</b> {source_id}<br>"

        if chunk_id != "":
            html += f"<b>块ID:</b> {chunk_id}<br>"

        html += f"<p>{content}</p>"
        html += "</div><br>"

    html += "</div>"
    return html


def format_process_info(process_lines):
    """格式化过程信息用于显示"""
    if not process_lines:
        return "未捕获到处理过程信息"

    return "\n".join(process_lines)


def format_detailed_process(detailed_info):
    """格式化详细的处理过程信息"""
    html = "<div class='detailed-process'>"

    # 添加总处理时间
    html += f"<div class='process-summary'><h4>📊 处理统计</h4>"
    html += f"<p>总处理时间: {detailed_info.get('总处理时间', '未知')}</p></div>"

    # 添加各阶段时间线
    stages = [
        ("竞赛识别阶段", "🔍", "competition-stage"),
        ("文档检索阶段", "📄", "document-stage"),
        ("LLM处理阶段", "🤖", "llm-stage")
    ]

    for stage_name, icon, class_name in stages:
        if stage_name in detailed_info and detailed_info[stage_name]:
            html += f"<div class='{class_name}'>"
            html += f"<h4>{icon} {stage_name}</h4>"
            html += "<div class='timeline'>"

            for entry in detailed_info[stage_name]:
                milestone_class = "milestone" if entry.get("is_milestone", False) else ""
                html += f"<div class='log-entry {milestone_class}'>"
                html += f"<span class='timestamp'>{entry['timestamp']}</span>"
                html += f"<span class='elapsed'>[+{entry['elapsed']}]</span>"
                html += f"<span class='message'>{entry['message']}</span>"
                html += "</div>"

            html += "</div></div>"

    # 添加详细日志
    if "详细日志" in detailed_info and detailed_info["详细日志"]:
        html += "<div class='all-logs'>"
        html += "<h4>📝 详细日志</h4>"
        html += "<details><summary>展开查看完整日志</summary>"
        html += "<div class='timeline'>"

        for entry in detailed_info["详细日志"]:
            milestone_class = "milestone" if entry.get("is_milestone", False) else ""
            html += f"<div class='log-entry {milestone_class}'>"
            html += f"<span class='timestamp'>{entry['timestamp']}</span>"
            html += f"<span class='elapsed'>[+{entry['elapsed']}]</span>"
            html += f"<span class='message'>{entry['message']}</span>"
            html += "</div>"

        html += "</div></details></div>"

    html += "</div>"
    return html


def main():
    # 页面标题
    st.markdown("<h1 class='main-header'>🤖 竞赛智能客服机器人</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>基于两阶段检索策略的智能问答系统，专业解答竞赛相关问题</p>",
                unsafe_allow_html=True)

    # 创建过程捕获处理器
    if "process_capture" not in st.session_state:
        st.session_state.process_capture = ProcessCaptureHandler()

    # 显示改进信息
    with st.expander("⭐ 系统说明", expanded=False):
        st.markdown("""
        ### 🔍 两阶段检索策略

        本系统采用了改进的两阶段检索策略，解决了传统RAG中信息片段化的问题：

        1. **第一阶段：竞赛识别** - 首先识别与问题最相关的竞赛
        2. **第二阶段：精准检索** - 在确定的竞赛范围内检索最相关的片段
        3. **结构化与概览增强** - 同时提供竞赛的结构化信息和概览
        4. **增强文档量** - 从相关竞赛中检索更多文档片段，提供更全面的信息

        这种方法确保了回答的连贯性和准确性，避免了不同竞赛信息的混淆，并通过增加参考文档数量提高回答的全面性。
        """)

    # 添加显示后台处理过程开关
    with st.sidebar:
        st.header("设置")
        show_process = st.toggle("显示后台处理过程", value=True, help="开启后可以查看系统检索和处理的详细过程")
        show_detailed_metrics = st.toggle("显示详细性能指标", value=True, help="开启后可以查看更详细的系统性能指标")

    # 侧边栏
    with st.sidebar:
        st.header("使用说明")
        st.markdown("""
        1. 在输入框中输入您关于竞赛的问题
        2. 系统将自动识别相关竞赛并给出答案
        3. 您可以查看来源信息了解详情
        4. 勾选"显示后台处理过程"可以查看系统如何工作
        5. 对话历史会自动保存

        **示例问题:**
        - 未来校园智能应用专项赛的报名时间是什么时候？
        - 3D编程模型创新设计专项赛的组织单位是谁？
        - 未来校园智能应用专项赛的参赛条件是什么？
        """)

        # 添加清空按钮
        if "messages" in st.session_state and st.button("清空聊天历史"):
            st.session_state.messages = []
            st.rerun()

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 如果是助手消息且包含来源信息，则显示来源信息
            if message["role"] == "assistant":
                if "sources" in message:
                    with st.expander("查看来源信息"):
                        if "query_time" in message:
                            st.markdown(f"**查询耗时:** {message['query_time']:.2f}秒")

                        st.markdown(format_sources(message["sources"]), unsafe_allow_html=True)

                # 如果有处理过程信息且开启了显示
                if "process_info" in message and show_process:
                    with st.expander("查看处理过程", expanded=False):
                        st.markdown("### 🔍 系统检索过程")
                        st.markdown("\n".join(message["process_info"]))

                # 如果有详细处理信息且开启了显示
                if "detailed_process" in message and show_detailed_metrics:
                    with st.expander("查看详细处理指标", expanded=False):
                        st.markdown("### 📊 系统性能指标", unsafe_allow_html=True)

                        # 显示性能指标卡片
                        if "performance_metrics" in message:
                            metrics = message["performance_metrics"]
                            st.markdown("<div class='process-metrics'>", unsafe_allow_html=True)

                            for key, value in metrics.items():
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <div>{key}</div>
                                    <div class='metric-value'>{value}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("</div>", unsafe_allow_html=True)

                        # 显示详细处理时间线
                        st.markdown(message["detailed_process"], unsafe_allow_html=True)

    # 初始化QA系统（如果尚未初始化）
    if "qa_system" not in st.session_state:
        with st.spinner("正在初始化问答系统..."):
            st.session_state.qa_system = initialize_qa_system()

    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        # 打印用户问题到终端
        print(f"\n用户问题: {prompt}")

        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 如果QA系统初始化成功，则调用回答
        if st.session_state.qa_system:
            with st.chat_message("assistant"):
                # 创建一个容器来显示流式文本
                answer_container = st.empty()

                # 开始捕获处理过程
                process_capture = st.session_state.process_capture
                process_capture.start_capture()

                try:
                    # 增加处理状态指示器
                    status_container = st.empty()
                    status_container.info("🔍 正在处理您的问题...")

                    # 创建流式处理器
                    stream_handler = StreamHandler(answer_container, process_capture)
                    # 创建详细信息处理器
                    verbose_handler = VerboseHandler(process_capture)

                    # 更新回调函数
                    st.session_state.qa_system.callbacks = [stream_handler, verbose_handler]

                    # 调用QA系统
                    result = st.session_state.qa_system.answer_question(prompt)

                    # 移除状态指示器
                    status_container.empty()

                    # 结束捕获处理过程
                    process_capture.end_capture()

                    # 获取答案和来源
                    answer = result["answer"]
                    sources = result["sources"]
                    elapsed_time = result["query_time"]

                    # 获取检索过程信息
                    process_info = process_capture.get_retrieval_process_info()
                    # 获取详细处理信息
                    detailed_process = process_capture.get_detailed_process_info()
                    # 获取性能指标
                    performance_metrics = process_capture.get_performance_metrics()

                    # 格式化详细处理信息为HTML
                    detailed_process_html = format_detailed_process(detailed_process)

                    # 添加机器人回答到历史，包含来源信息和处理过程
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "query_time": elapsed_time,
                        "process_info": process_info,
                        "detailed_process": detailed_process_html,
                        "performance_metrics": performance_metrics
                    })

                    # 显示来源信息
                    with st.expander("查看来源信息"):
                        st.markdown(f"**查询耗时:** {elapsed_time:.2f}秒")
                        st.markdown(format_sources(sources), unsafe_allow_html=True)

                    # 如果开启了显示处理过程
                    if show_process:
                        with st.expander("查看处理过程", expanded=True):
                            st.markdown("<div class='process-display'>", unsafe_allow_html=True)

                            # 分段显示处理信息
                            process_html = ""
                            current_stage = None

                            for line in process_info:
                                # 检测阶段标题
                                if line.startswith("**🔍 阶段1"):
                                    current_stage = "stage1"
                                    # 移除Markdown标记并展示为HTML
                                    clean_line = line.replace("**", "")
                                    process_html += f"<p class='process-stage'>{clean_line}</p>"
                                elif line.startswith("**📄 阶段2"):
                                    current_stage = "stage2"
                                    clean_line = line.replace("**", "")
                                    process_html += f"<p class='process-stage'>{clean_line}</p>"
                                elif line.startswith("**✓ 检索完成"):
                                    current_stage = "closing"
                                    clean_line = line.replace("**", "")
                                    process_html += f"<p class='process-stage'>{clean_line}</p>"
                                elif line.strip() == "":
                                    # 空行转换为HTML空行
                                    process_html += "<br>"
                                else:
                                    # 普通行，根据格式化添加样式
                                    if line.startswith("**"):
                                        # 重要信息（加粗）
                                        clean_line = line.replace("**", "")
                                        process_html += f"<p><b>{clean_line}</b></p>"
                                    elif line.startswith("* "):
                                        # 列表项
                                        clean_line = line[2:]
                                        process_html += f"<p class='process-result'>• {clean_line}</p>"
                                    else:
                                        # 普通文本
                                        process_html += f"<p>{line}</p>"

                            st.markdown(process_html + "</div>", unsafe_allow_html=True)

                    # 如果开启了显示详细性能指标
                    if show_detailed_metrics:
                        with st.expander("查看详细处理指标", expanded=True):
                            st.markdown("### 📊 系统性能指标", unsafe_allow_html=True)

                            # 显示性能指标卡片
                            st.markdown("<div class='process-metrics'>", unsafe_allow_html=True)

                            for key, value in performance_metrics.items():
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <div>{key}</div>
                                    <div class='metric-value'>{value}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("</div>", unsafe_allow_html=True)

                            # 显示详细处理时间线
                            st.markdown(detailed_process_html, unsafe_allow_html=True)

                except Exception as e:
                    # 结束捕获处理过程
                    process_capture.end_capture()

                    error_message = f"很抱歉，处理您的问题时出现了错误: {str(e)}"
                    st.error(error_message)
                    # 在终端输出错误信息
                    print(f"错误: {str(e)}", file=sys.stderr)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "process_info": process_capture.get_retrieval_process_info()
                    })
        else:
            with st.chat_message("assistant"):
                error_message = "很抱歉，问答系统未成功初始化，请确保已运行 `python 3_knowledge_base.py` 构建知识库。"
                st.error(error_message)
                print(error_message, file=sys.stderr)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()