# Use official python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data folders inside container
RUN mkdir -p data/raw data/combined

# Default command
CMD ["python", "src/fetch_stock_data.py"]