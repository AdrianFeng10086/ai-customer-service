import pdfplumber
import os
import re
import sys
import time

# === DOCTR CODE START ===
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# 初始化一个 Doctr OCR 模型（用的预训练权重，包含文本检测+识别）
model = ocr_predictor(pretrained=True)
# === DOCTR CODE END ===


CONFIG = {
    "min_page_text_length": 10,  # 提取文本长度低于此值时，视为“未提取到有效文本”，从而转而使用OCR
    "x_tolerance": 3,
    "y_tolerance": 3,
    "batch_print_pages": 5
}


def extract_text_from_pdf(pdf_path):
    raw_text = ""
    start_time = time.time()

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  - PDF共有 {total_pages} 页")

            empty_pages = 0
            for i, page in enumerate(pdf.pages, 1):
                # 显示处理进度
                if total_pages > 10 and i % CONFIG["batch_print_pages"] == 0:
                    progress = (i / total_pages) * 100
                    print(f"  - 正在处理第 {i}/{total_pages} 页... ({progress:.1f}%)")

                # 1) 先用 pdfplumber 提取文本
                page_text = page.extract_text(
                    x_tolerance=CONFIG["x_tolerance"],
                    y_tolerance=CONFIG["y_tolerance"]
                )

                # 2) 若文字不足，使用 Doctr OCR
                if not page_text or len(page_text.strip()) < CONFIG["min_page_text_length"]:
                    # 将 PDF 页面转换为图像 (PIL格式)，提高分辨率可改善识别效果
                    page_image = page.to_image(resolution=300).original  # PIL Image

                    # === DOCTR CODE START ===
                    # 构建一个 DocumentFile 对象；doctr支持直接从 PIL.Image / ndarray 等读取
                    doc = DocumentFile.from_images([page_image])
                    # 使用加载好的 model 进行推理，得到识别结果
                    result = model(doc)

                    # 将识别后的页面对象遍历，提取纯文本
                    ocr_text_list = []
                    for page_res in result.pages:
                        # blocks -> lines -> words
                        for block in page_res.blocks:
                            for line in block.lines:
                                # 将该行所有单词拼接为字符串
                                line_text = " ".join(word.value for word in line.words)
                                ocr_text_list.append(line_text)
                    # 用换行符把所有行拼接起来
                    ocr_text = "\n".join(ocr_text_list)
                    # === DOCTR CODE END ===

                    # 如果OCR得到的文字依然不足，视为空页
                    if ocr_text and len(ocr_text.strip()) >= CONFIG["min_page_text_length"]:
                        page_text = ocr_text
                    else:
                        page_text = ""

                # 3) 判断最终文本是否有效
                if page_text and len(page_text.strip()) > CONFIG["min_page_text_length"]:
                    raw_text += f"\n\n===== 第 {page.page_number} 页 =====\n\n"
                    raw_text += page_text + "\n"
                else:
                    empty_pages += 1

            if empty_pages > 0:
                print(f"  - 注意: 检测到 {empty_pages} 个页面未提取到有效文本")

    except Exception as e:
        print(f"  - 打开或读取PDF时出错: {str(e)}")
        return ""

    # 文本后处理
    print("  - 正在进行文本清理...")
    text = clean_text(raw_text)

    elapsed_time = time.time() - start_time
    print(f"  - 文本提取和清理完成! 用时: {elapsed_time:.2f} 秒")
    print(f"  - 提取文本总长度: {len(text)} 字符")
    return text


def clean_text(text):
    """
    对文本进行简单清理：去除多余空格、修复断行、移除控制字符等
    """
    if not text:
        return ""

    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'([^\n\.,;!?。，；！？\s])[ \t]*\n[ \t]*([^\n])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\x00-\x09\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'(?m)^[ \t]+', '', text)
    text = re.sub(r'\n*===== 第 (\d+) 页 =====\n*', r'\n\n===== 第 \1 页 =====\n\n', text)

    return text


def main():
    print("="*50)
    print("PDF文本提取工具（基于 pdfplumber + doctr OCR）")
    print("="*50)

    data_dir = os.path.join(os.getcwd(), "data")

    output_dir = os.path.join(os.getcwd(), "extracted")

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 不存在!")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 搜索 data 目录下的所有 PDF
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"警告: 在 '{data_dir}' 目录中未找到PDF文件!")
        sys.exit(0)

    print(f"找到 {len(pdf_files)} 个PDF文件需要处理")
    print("-"*50)

    success_count = 0
    error_count = 0
    start_time = time.time()

    # 逐个处理 PDF
    for i, pdf_filename in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] 正在处理: {pdf_filename}")

        try:
            pdf_path = os.path.join(data_dir, pdf_filename)
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            print(f"  - 文件大小: {file_size_mb:.2f} MB")

            file_start_time = time.time()
            extracted_text = extract_text_from_pdf(pdf_path)
            file_elapsed_time = time.time() - file_start_time

            if not extracted_text:
                print(f"  - 警告: 未能从 {pdf_filename} 提取到文本内容")
                error_count += 1
                continue

            # 构建输出 txt 文件路径
            base_name = os.path.splitext(pdf_filename)[0]
            output_filename = f"extracted_{base_name}.txt"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            print(f"  - 成功：已保存文本到: {output_filename}")
            print(f"  - 处理用时: {file_elapsed_time:.2f} 秒")
            success_count += 1

        except Exception as e:
            print(f"  - 错误：处理 {pdf_filename} 时出错: {str(e)}")
            error_count += 1

        print("-"*50)

    total_elapsed_time = time.time() - start_time
    print(f"处理完成！成功: {success_count} 个, 失败: {error_count} 个")
    print(f"总处理时间: {total_elapsed_time:.2f} 秒")

    if success_count > 0:
        print(f"提取的文本文件已保存到: {output_dir} 目录")


if __name__ == "__main__":
    main()
