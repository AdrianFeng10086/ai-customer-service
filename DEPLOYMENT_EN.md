# Deployment and Configuration Guide

## 🔐 API Key Configuration

For security reasons, all API keys have been replaced with `x`. Configuration is required before actual use.

### Step 1: Copy Environment Variable Template

```bash
cp .env.example .env
```

### Step 2: Configure API Keys

Edit the `.env` file and fill in your API keys:

```bash
# HuggingFace API Configuration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-reasoner
```

### Step 3: Use Environment Variables in Code

To load API keys from environment variables, use the following approach:

```python
from dotenv import load_dotenv
import os

load_dotenv()

API_CONFIG = {
    "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
    "openai_api_key": os.getenv("DEEPSEEK_API_KEY"),
}
```

## 📦 Obtaining API Keys

### HuggingFace API

1. Visit [HuggingFace Official Website](https://huggingface.co)
2. Register or log in to your account
3. Go to Settings → Access Tokens
4. Create a new API Token
5. Copy the Token to the `.env` file

### DeepSeek API

1. Visit [DeepSeek Official Website](https://platform.deepseek.com)
2. Register or log in to your account
3. Go to API Keys management
4. Create a new API Key
5. Copy the Key to the `.env` file

## 🚀 Quick Deployment

### Local Development Environment

```bash
# 1. Clone the project
git clone <repository-url>
cd <project-directory>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit the .env file and fill in API keys

# 5. Run the application
streamlit run app.py
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
# Build image
docker build -t competition-qa-bot .

# Run container
docker run -p 8501:8501 \
  -e HUGGINGFACE_API_KEY="your_key" \
  -e DEEPSEEK_API_KEY="your_key" \
  competition-qa-bot
```

## 🔒 Security Best Practices

1. **Never commit API keys** - Use `.env` files and `.gitignore`
2. **Use environment variables** - Prioritize environment variables in production
3. **Regularly rotate keys** - Regularly update and rotate API keys
4. **Limit permission scope** - Set minimum necessary permissions at API provider
5. **Monitor usage** - Regularly check API usage logs

## 📋 Environment Variable Description

| Variable Name | Description | Required |
|--------------|-------------|----------|
| HUGGINGFACE_API_KEY | HuggingFace API key | ✅ |
| HUGGINGFACE_EMBEDDING_MODEL | Embedding model name | ✅ |
| DEEPSEEK_API_KEY | DeepSeek API key | ✅ |
| DEEPSEEK_BASE_URL | DeepSeek API base URL | ✅ |
| DEEPSEEK_MODEL | DeepSeek model name | ✅ |
| KNOWLEDGE_BASE_DIR | Knowledge base directory | ❌ |
| DATA_DIR | Data file directory | ❌ |
| EXTRACTED_DIR | Extracted file directory | ❌ |
| RESULT_DIR | Result output directory | ❌ |
| LOG_LEVEL | Log level | ❌ |
| DEBUG | Debug mode | ❌ |

## 🐛 Troubleshooting

### Issue: Invalid API Key
- Check if the key in the `.env` file is correct
- Ensure the API key has not expired
- Check if API quota is sufficient

### Issue: Module Import Error
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Database Connection Failed
```bash
# Check knowledge base directory permissions
ls -la knowledge_base/

# Reinitialize knowledge base
python 3_knowledge.py
```

---

**Last Updated**: January 2026
