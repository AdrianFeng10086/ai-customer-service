# 部署和配置指南

## 🔐 API密钥配置

由于安全原因，所有API密钥已被替换为 `x`。在实际使用前需要进行配置。

### 步骤1：复制环境变量模板

```bash
cp .env.example .env
```

### 步骤2：配置API密钥

编辑 `.env` 文件，填入你的API密钥：

```bash
# HuggingFace API Configuration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-reasoner
```

### 步骤3：在代码中使用环境变量

如需从环境变量加载API密钥，可以使用以下方式：

```python
from dotenv import load_dotenv
import os

load_dotenv()

API_CONFIG = {
    "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
    "openai_api_key": os.getenv("DEEPSEEK_API_KEY"),
}
```

## 📦 获取API密钥

### HuggingFace API

1. 访问 [HuggingFace官网](https://huggingface.co)
2. 注册或登录账户
3. 进入 Settings → Access Tokens
4. 创建新的 API Token
5. 复制 Token 到 `.env` 文件

### DeepSeek API

1. 访问 [DeepSeek官网](https://platform.deepseek.com)
2. 注册或登录账户
3. 进入 API Keys 管理
4. 创建新的 API Key
5. 复制 Key 到 `.env` 文件

## 🚀 快速部署

### 本地开发环境

```bash
# 1. 克隆项目
git clone <repository-url>
cd <project-directory>

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 5. 运行应用
streamlit run app.py
```

### Docker部署

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

构建和运行：

```bash
# 构建镜像
docker build -t competition-qa-bot .

# 运行容器
docker run -p 8501:8501 \
  -e HUGGINGFACE_API_KEY="your_key" \
  -e DEEPSEEK_API_KEY="your_key" \
  competition-qa-bot
```

## 🔒 安全最佳实践

1. **永远不要提交API密钥** - 使用 `.env` 文件和 `.gitignore`
2. **使用环境变量** - 在生产环境中优先使用环境变量
3. **定期轮换密钥** - 定期更新和轮换API密钥
4. **限制权限范围** - 在API提供商设置最小必要权限
5. **监控使用情况** - 定期检查API使用日志

## 📋 环境变量说明

| 变量名 | 说明 | 必需 |
|--------|------|------|
| HUGGINGFACE_API_KEY | HuggingFace API密钥 | ✅ |
| HUGGINGFACE_EMBEDDING_MODEL | 嵌入模型名称 | ✅ |
| DEEPSEEK_API_KEY | DeepSeek API密钥 | ✅ |
| DEEPSEEK_BASE_URL | DeepSeek API基础URL | ✅ |
| DEEPSEEK_MODEL | DeepSeek模型名称 | ✅ |
| KNOWLEDGE_BASE_DIR | 知识库目录 | ❌ |
| DATA_DIR | 数据文件目录 | ❌ |
| EXTRACTED_DIR | 提取文件目录 | ❌ |
| RESULT_DIR | 结果输出目录 | ❌ |
| LOG_LEVEL | 日志级别 | ❌ |
| DEBUG | 调试模式 | ❌ |

## 🐛 故障排除

### 问题：API密钥无效
- 检查 `.env` 文件中的密钥是否正确
- 确保API密钥未过期
- 检查API额度是否充足

### 问题：模块导入错误
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt
```

### 问题：数据库连接失败
```bash
# 检查知识库目录权限
ls -la knowledge_base/

# 重新初始化知识库
python 3_knowledge.py
```
---

**最后更新**: 2026年1月
