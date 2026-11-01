# Dockerfile for Distributed Bias Mitigation on Apache Spark
FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Java 21, Python 3, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

# Copy project source
COPY . .

RUN chmod +x run_pipeline.sh

# Default command runs quick pipeline demo
CMD ["./run_pipeline.sh", "--quick"]
