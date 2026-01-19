"""
竞赛信息抽取脚本
功能：
1. 读取 extracted/ 目录下的 txt 文件
2. 调用大模型（DeepSeek Chat）或正则表达式提取六大核心字段
3. 结果输出到 result/result_1.xlsx
"""

import os                  # 操作系统功能
import sys                 # 系统功能
import time                # 时间处理
import re                  # 正则表达式
import json                # JSON 处理
import datetime            # 日期时间处理
import pandas as pd        # 数据分析
from openai import OpenAI  # OpenAI/DeepSeek API 客户端

# ---------------------- API 配置 ----------------------
API_CONFIG = {
    "name": "DeepSeek Chat",
    "api_key": "x",   # DeepSeek API 密钥
    "base_url": "https://api.deepseek.com",             # DeepSeek 代理地址
    "model": "deepseek-chat"                            # DeepSeek 模型名称
}
# -----------------------------------------------------


# -----------------------------------------------------
# 文本预处理：长文本裁剪 + 关键词突出
# -----------------------------------------------------
def preprocess_text(text: str, max_tokens: int = 4000) -> str:
    """
    - 去掉页脚页眉等冗余
    - 超长文本仅保留前三分之一和后 1/3，并在中间加“关键信息”段
    """
    text = re.sub(r'===== 第 \d+ 页 =====', '', text)   # 示例页标
    lines = text.split('\n')

    if len(text) < 15000:      # <≈3000 tokens 不做截断
        return text

    start = '\n'.join(lines[: len(lines) // 3])
    end   = '\n'.join(lines[-len(lines) // 3:])
    middle = '\n'.join(lines[len(lines) // 3 : -len(lines) // 3])

    key_patterns = [
        r'(?:报名时间|报名日期|报名截止|registration)[：:\s]+([^。\n]+)',
        r'(?:比赛时间|竞赛时间|决赛时间|选拔赛时间)[：:\s]+([^。\n]+)',
        r'(?:官方网站|官网|网站)[：:\s]+(http[s]?://[^\s]+)',
        r'(?:主办单位|承办单位|组织单位)[：:\s]+([^。\n]+)'
    ]
    key_info = []
    for pat in key_patterns:
        key_info += re.findall(pat, middle, re.I)

    return (
        start +
        "\n\n关键信息:\n" + "\n".join(key_info) +
        "\n\n" + end
    )


# =====================================================
#             ↓↓↓ 正则字段提取核心逻辑 ↓↓↓
# =====================================================

# ===== 修改开始：发布时间提取 =====
# ===== 再次修改：发布时间提取 (放宽年限 + 新日期格式) =====
def _extract_publish_date(text: str, year_span: int = 10) -> str:
    """
    ① 先匹配含“发布时间 / 发布日期 / 发布”关键词的显式描述
    ② 若失败，在（当前年‑year_span）~当前年区间内，取全文出现的最早日期
       日期写法支持：
          - 2024年4月15日 / 2024年4月
          - 2024‑04‑15 / 2024‑04
          - 2024.04.15 / 2024.04
    返回统一格式：YYYY年M月[可选D日]；若未找到则返回“未提供”
    """
    # 1. 关键词 + 中文年月日
    kw_patterns = [
        r'(?:发布时间|发布日期|发布于|发布)[：:\s]*(\d{4})[年\-.](\d{1,2})[月\-.]?(?:\s*(\d{1,2}))[日号]?',
        r'(\d{4})[年\-.](\d{1,2})[月\-.]?(?:\s*(\d{1,2}))[日号]?\s*发布'
    ]
    for pat in kw_patterns:
        m = re.search(pat, text)
        if m:
            y, mn, d = m.group(1), m.group(2), m.group(3) or ''
            return f"{y}年{int(mn)}月{f'{int(d)}日' if d else ''}"

    # 2. 全局日期搜集（近 year_span 年）
    #    支持中文“年/月/日”，也支持 - . /
    date_regex = r'(\d{4})[年\-.\/](\d{1,2})[月\-.\/]?(?:\s*(\d{1,2}))?[日号]?'
    dates = re.findall(date_regex, text)
    if dates:
        cur_year = datetime.datetime.now().year
        valid = [
            (int(y), int(m), int(d) if d else 0)
            for y, m, d in dates
            if cur_year - year_span <= int(y) <= cur_year      # 放宽到 year_span
        ]
        if valid:
            valid.sort()                                       # 取最早
            y, mn, d = valid[0]
            return f"{y}年{mn}月{f'{d}日' if d else ''}"

    return "未提供"
# ===== 修改结束 =====

# ===== 修改开始：赛道提取 =====
def _extract_tracks(text: str) -> str:
    """
    三步提取赛道 / 组别 / 方向：
    ① “赛道：XXX、YYY” ② “设置/分为/包含…赛道” ③ 列表行枚举
    """
    m = re.search(r'(?:赛道|组别|方向)[：:]\s*([\u4e00-\u9fa5A-Za-z0-9、，/\\（）() &]{3,80})', text)
    raw = m.group(1) if m else ''
    if not raw:
        m = re.search(
            r'(?:设置|分为|包含)[^。\n]{0,20}?(?:赛道|组别|方向)[：:\s，]*([\u4e00-\u9fa5A-Za-z0-9、，/\\（）() &]{3,80})',
            text
        )
        raw = m.group(1) if m else ''
    if raw:
        return re.sub(r'[、，/,]+', '；', raw.strip(' 。\n'))

    tracks = []
    for line in text.splitlines():
        m = re.match(r'^\s*[（(]?\d+[)）.、]\s*([\u4e00-\u9fa5A-Za-z0-9（）() &]{2,40})(?:赛道|组别|方向)', line)
        if m:
            tracks.append(m.group(1))
    if tracks:
        return '；'.join(dict.fromkeys(tracks))

    return "未提供"
# ===== 修改结束 =====


def extract_basic_info_with_regex(text: str) -> dict:
    """
    使用纯正则从全文中提取六字段。
    """
    info = {
        "赛事名称": "未提供",
        "赛道": "未提供",
        "发布时间": "未提供",
        "报名时间": "未提供",
        "组织单位": "未提供",
        "官网": "未提供"
    }

    # 1) 赛事名称
    title_pats = [
        r'(^|\n)第[一二三四五六七八九十\d]+届.*?(?:比赛|竞赛|大赛|专项赛|挑战赛|邀请赛)[^\n]*',
        r'(^|\n).{2,60}?(?:比赛|竞赛|大赛|专项赛|挑战赛|邀请赛)[^\n]*'
    ]
    for pat in title_pats:
        m = re.search(pat, text)
        if m:
            info["赛事名称"] = m.group(0).strip().replace('\n', '')
            break

    # 2) 赛道
    info["赛道"] = _extract_tracks(text)

    # 3) 发布时间
    info["发布时间"] = _extract_publish_date(text)

    # 4) 报名时间
    reg_pats = [
        r'报名(?:时间|日期|起止日期|期限)[：:\s]*([^\n。；;]{4,60})',
        r'报名时间为[：:\s]*([^\n。；;]{4,60})'
    ]
    for pat in reg_pats:
        m = re.search(pat, text)
        if m:
            info["报名时间"] = m.group(1).strip()
            break

    # 5) 官网
    web_pats = [
        r'(?:官方网站|官网|网址|网站)[：:\s]*(http[s]?://[^\s，,。；;]+)',
        r'(http[s]?://[^\s，,。；;]+)\s*(?:为)?\s*(?:官方网站|官网)'
    ]
    for pat in web_pats:
        m = re.search(pat, text)
        if m:
            info["官网"] = m.group(1).strip()
            break

    # 6) 组织单位
    org_pats = [
        r'(?:主办单位|主办方)[：:\s]*([^\n。；;]{3,120})',
        r'(?:承办单位|承办方)[：:\s]*([^\n。；;]{3,120})',
        r'(?:组织单位|组织方)[：:\s]*([^\n。；;]{3,120})'
    ]
    orgs = []
    for pat in org_pats:
        m = re.search(pat, text)
        if m:
            orgs.append(m.group(1).strip())
    if orgs:
        info["组织单位"] = "；".join(dict.fromkeys(orgs))

    return info
# =====================================================


def extract_competition_info(text: str, max_retries: int = 3) -> dict:
    """
    先用 LLM 提取，失败则降级到正则提取。
    """
    processed = preprocess_text(text)

    prompt = f"""
    请从以下竞赛文档中提取信息并以 JSON 返回（字段用中文）：
    1. 赛事名称
    2. 赛道
    3. 发布时间 (格式示例：2024年4月)
    4. 报名时间
    5. 组织单位
    6. 官网
    未知请填“未提供”，只返回 JSON，无额外内容。

    文档：
    {processed}
    """

    for attempt in range(max_retries):
        try:
            print(f"  - 调用 {API_CONFIG['name']} (尝试 {attempt + 1}/{max_retries})")
            client = OpenAI(
                api_key=API_CONFIG["api_key"],
                base_url=API_CONFIG["base_url"]
            )
            resp = client.chat.completions.create(
                model=API_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content.strip()
            # 仅截取 JSON
            m = re.search(r'(\{.*?\})', content, re.S)
            content = m.group(1) if m else content
            ai_result = json.loads(content)

            # 缺失字段用正则补
            regex_result = extract_basic_info_with_regex(text)
            final = {}
            for field in ["赛事名称", "赛道", "发布时间", "报名时间", "组织单位", "官网"]:
                final[field] = (
                    ai_result.get(field) if field in ai_result and ai_result[field] != "未提供"
                    else regex_result.get(field, "未提供")
                )
            return final

        except json.JSONDecodeError as e:
            print(f"  - JSON 解析失败：{e}")
        except Exception as e:
            print(f"  - API 调用异常：{e}")

        time.sleep(1)  # 重试间隔

    print("  - 使用正则兜底提取")
    return extract_basic_info_with_regex(text)


def main() -> None:
    """批处理 extracted/*.txt 并输出 Excel"""
    print("=" * 50)
    print("竞赛信息提取程序")
    print("=" * 50)

    result_dir = os.path.join(os.getcwd(), "result")
    extracted_dir = os.path.join(os.getcwd(), "extracted")

    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(extracted_dir):
        print(f"错误：目录 {extracted_dir} 不存在")
        sys.exit(1)

    txt_files = [f for f in os.listdir(extracted_dir) if f.lower().endswith('.txt')]
    if not txt_files:
        print("未找到待处理的 TXT 文件")
        sys.exit(0)

    print(f"共发现 {len(txt_files)} 个 TXT 文件")
    print("-" * 50)

    results = []
    for idx, fname in enumerate(txt_files, 1):
        print(f"[{idx}/{len(txt_files)}] 处理 {fname}")
        try:
            with open(os.path.join(extracted_dir, fname), 'r', encoding='utf-8') as f:
                text = f.read()

            info = extract_competition_info(text)
            info["源文件"] = fname
            results.append(info)

            for k, v in info.items():
                if k != "源文件":
                    print(f"    {k}: {v}")

        except Exception as e:
            print(f"  - 处理失败：{e}")
            results.append({
                "赛事名称": "处理错误",
                "赛道": "未提供",
                "发布时间": "未提供",
                "报名时间": "未提供",
                "组织单位": "未提供",
                "官网": "未提供",
                "源文件": fname,
                "错误信息": str(e)
            })
        print("-" * 50)

    if results:
        df = pd.DataFrame(results)
        columns = ["赛事名称", "赛道", "发布时间", "报名时间", "组织单位", "官网"]
        df = df[[c for c in columns if c in df.columns]]
        out_path = os.path.join(result_dir, "result_1.xlsx")
        df.to_excel(out_path, index=False)
        print(f"已输出 {out_path} (共 {len(results)} 条)")
    else:
        print("没有生成任何结果")

# ------------------ 主入口 ------------------
if __name__ == "__main__":
    main()
