import os
import shutil
import argparse
import json
import hashlib
from pathlib import Path
import importlib.util as iu

# === 路径配置 ================================================================
ROOT_DIR      = Path(__file__).resolve().parent
PDF_DIR       = ROOT_DIR / "newdata"
EXTRACT_DIR   = ROOT_DIR / "extracted"
RESULT_DIR    = ROOT_DIR / "result"
DB_DIR        = ROOT_DIR / "knowledge_base"
META_FILE     = ROOT_DIR / "processed_files.json"

PDF_EXTRACTOR_MOD   = ROOT_DIR / "1_pdf2txt.py"
TXT_EXTRACTION_MOD  = ROOT_DIR / "2_txt_extraction.py"
KNOWLEDGE_BASE_MOD  = ROOT_DIR / "3_knowledge_base.py"

# === 动态导入模块 ===========================================================
def load_mod(path: Path, name: str):
    spec = iu.spec_from_file_location(name, path)
    mod  = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pdf_extractor   = load_mod(PDF_EXTRACTOR_MOD,  "pdf_extractor")
text_extraction = load_mod(TXT_EXTRACTION_MOD, "text_extraction")
kb_builder      = load_mod(KNOWLEDGE_BASE_MOD, "knowledge_base")

# === 工具函数 ================================================================
def compute_hash(path: Path) -> str:
    """计算文件 SHA256"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_metadata() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {}

def save_metadata(meta: dict):
    META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

def ensure_dirs():
    """只清空 EXTRACT_DIR 和 RESULT_DIR，DB_DIR 保留旧数据"""
    for d in (EXTRACT_DIR, RESULT_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

# === 增量更新流程 ===========================================================
def update_knowledge_base(verbose: bool = True):
    # 0. 准备目录 & 元数据
    ensure_dirs()
    old_meta = load_metadata()
    new_meta = {}

    # 1. 扫描所有 PDF，分类
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"在 {PDF_DIR} 未找到 PDF 文件")

    changed = []
    all_names = [p.name for p in pdf_files]
    removed = set(old_meta.keys()) - set(all_names)

    for pdf in pdf_files:
        name = pdf.name
        h = compute_hash(pdf)
        new_meta[name] = h
        if name not in old_meta or old_meta[name] != h:
            changed.append(pdf)
        # else: 未变更

    # 2. 清理已删除 & 变更文件在知识库中的旧数据
    for name in removed:
        if verbose:
            print(f"[0] 删除已移除文件的数据：{name}")
        kb_builder.remove_file_data(name, DB_DIR)

    for pdf in changed:
        if verbose:
            print(f"[0] 清理变更文件的旧数据：{pdf.name}")
        kb_builder.remove_file_data(pdf.name, DB_DIR)

    # 3. 仅对新增/变更文件做 PDF→TXT
    for pdf in changed:
        if verbose:
            print(f"[1/4] 抽取文本：{pdf.name}")
        text = pdf_extractor.extract_text_from_pdf(pdf)
        out = EXTRACT_DIR / f"extracted_{pdf.stem}.txt"
        out.write_text(text, encoding="utf-8")

    # 4. 对新增/变更的 TXT 生成/更新 Excel
    if changed:
        if verbose:
            print(f"[2/4] 信息抽取生成 Excel（仅追加新增/变更）…")
        txt_paths = [EXTRACT_DIR / f"extracted_{p.stem}.txt" for p in changed]
        text_extraction.extract_from_txt(txt_paths)
    else:
        if verbose:
            print("无新增或变更，跳过 TXT→Excel 步骤。")

    excel_path = RESULT_DIR / "result_1.xlsx"
    if not excel_path.exists() and changed:
        raise FileNotFoundError("未生成 result_1.xlsx，请检查 2_txt_extraction.py")

    # 5. 向量化处理（仅对新增/变更）
    if changed:
        if verbose:
            print(f"[3/4] 构建/更新向量数据库（仅新增/变更）…")
        embeddings = kb_builder.setup_embedding_model()
        # 假设 process_excel_data 可接受 only_for 参数
        kb_builder.process_excel_data(excel_path, embeddings, DB_DIR,
                                      only_for=[p.name for p in changed])
    else:
        if verbose:
            print("无新增或变更，跳过向量化步骤。")

    # 6. 保存新元数据
    save_metadata(new_meta)
    if verbose:
        print("🎉 知识库增量更新完成！")

# === CLI ===================================================================
def main():
    parser = argparse.ArgumentParser(description="增量更新知识库")
    parser.add_argument("--quiet", action="store_true", help="不打印详细日志")
    args = parser.parse_args()

    try:
        update_knowledge_base(verbose=not args.quiet)
    except Exception as e:
        print(f"❌ 过程失败：{e}")
        raise

if __name__ == "__main__":
    main()
