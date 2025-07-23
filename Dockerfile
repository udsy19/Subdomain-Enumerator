# Ultra-Robust Subdomain Enumerator - Docker Container
FROM python:3.11-slim

LABEL maintainer="Security Team"
LABEL description="Ultra-Robust Subdomain Enumerator with Beautiful TUI"
LABEL version="3.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libc6-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY tui_main.py main_tui.py main.py ./
COPY wordlists/ ./wordlists/
COPY README.md TECHNICAL_README.md ./

# Create output directory
RUN mkdir -p output

# Make TUI executable
RUN chmod +x tui_main.py

# Set up environment
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm-256color

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import tui_main; print('OK')" || exit 1

# Default command
CMD ["python3", "tui_main.py"]

# Usage examples:
# Build: docker build -t ultra-subdomain-enumerator .
# Run:   docker run -it --rm -v $(pwd)/output:/app/output ultra-subdomain-enumerator
# Demo:  docker run -it --rm ultra-subdomain-enumerator python3 tui_main.py --demo