# 1. Start with a lightweight Python 3.11 environment
FROM python:3.11-slim

# 2. Tell the server to do all its work inside a folder called /app
WORKDIR /app

# 3. Copy your requirements file first
COPY requirements.txt .

# 4. Install all your Python libraries 
# (We use --no-cache-dir to keep the server lightweight)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your code and your AI models into the server
COPY . .

# 6. Expose the port that FastAPI uses
EXPOSE 8000

# 7. The command to turn the server on
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]