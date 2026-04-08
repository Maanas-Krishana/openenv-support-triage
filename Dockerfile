FROM python:3.10-slim

WORKDIR /app

# Install dependencies (requires openenv)
RUN pip install --no-cache-dir openenv pydantic uvicorn fastapi

# Copy files
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
