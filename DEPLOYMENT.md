# PaddleOCR Integration Deployment Guide

This guide provides comprehensive instructions for deploying the PaddleOCR integration service in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB free space (for models and temp files)
- **CPU**: 2+ cores recommended
- **GPU**: Optional (CUDA-compatible for acceleration)

### Software Dependencies
- **Python**: 3.9 or higher
- **Docker**: 20.10+ (optional)
- **Git**: For cloning the repository

## Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/zbest1000/CalcPack.git
cd CalcPack

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### 2. Start Service
```bash
# Method 1: Direct Python
python3 paddleocr_service.py serve

# Method 2: Docker
docker-compose up -d

# Method 3: Background service
nohup python3 paddleocr_service.py serve > logs/service.log 2>&1 &
```

### 3. Test Installation
```bash
# Run test script
python3 examples/test_ocr.py --test all

# Check health
curl http://localhost:8000/health
```

## Local Development

### Environment Setup
```bash
# Create virtual environment
python3 -m venv paddleocr-env
source paddleocr-env/bin/activate  # Linux/macOS
# paddleocr-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings
```

### Development Server
```bash
# Start with hot reload
python3 paddleocr_service.py serve --reload

# Start with custom configuration
python3 paddleocr_service.py serve --host 0.0.0.0 --port 8080 --workers 2
```

### MCP Server Development
```bash
# Start MCP server in development mode
python3 mcp_server.py --service-url http://localhost:8000
```

## Docker Deployment

### Single Container
```bash
# Build image
docker build -t paddleocr-service .

# Run container
docker run -d \
  --name paddleocr-service \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -e PADDLE_OCR_LOG_LEVEL=INFO \
  paddleocr-service
```

### Multi-Container with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale paddleocr-service=3

# Stop services
docker-compose down
```

### Docker Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  paddleocr-service:
    environment:
      - PADDLE_OCR_ENABLE_GPU=true
      - PADDLE_OCR_CPU_THREADS=8
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## Production Deployment

### System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
  python3 python3-pip python3-venv \
  nginx supervisor \
  curl wget git

# Create service user
sudo useradd -r -s /bin/false paddleocr
sudo mkdir -p /opt/paddleocr
sudo chown paddleocr:paddleocr /opt/paddleocr
```

### Application Deployment
```bash
# Deploy application
sudo -u paddleocr git clone https://github.com/zbest1000/CalcPack.git /opt/paddleocr
cd /opt/paddleocr

# Install dependencies
sudo -u paddleocr python3 -m venv venv
sudo -u paddleocr ./venv/bin/pip install -r requirements.txt

# Configure environment
sudo -u paddleocr cp .env.example .env
# Edit production settings in .env
```

### Systemd Service
```bash
# Create systemd service file
sudo tee /etc/systemd/system/paddleocr.service > /dev/null <<EOF
[Unit]
Description=PaddleOCR Service
After=network.target

[Service]
Type=simple
User=paddleocr
Group=paddleocr
WorkingDirectory=/opt/paddleocr
Environment=PATH=/opt/paddleocr/venv/bin
ExecStart=/opt/paddleocr/venv/bin/python paddleocr_service.py serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable paddleocr
sudo systemctl start paddleocr
sudo systemctl status paddleocr
```

### Nginx Configuration
```bash
# Create nginx configuration
sudo tee /etc/nginx/sites-available/paddleocr > /dev/null <<EOF
upstream paddleocr_backend {
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://paddleocr_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Increase timeouts for OCR processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 120s;
        
        # Increase max body size for image uploads
        client_max_body_size 10M;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://paddleocr_backend/health;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/paddleocr/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/paddleocr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL Configuration (Let's Encrypt)
```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Configuration

### Environment Variables
```bash
# Service settings
export PADDLE_OCR_HOST=0.0.0.0
export PADDLE_OCR_PORT=8000
export PADDLE_OCR_WORKERS=4
export PADDLE_OCR_LOG_LEVEL=INFO

# Performance settings
export PADDLE_OCR_ENABLE_GPU=false
export PADDLE_OCR_CPU_THREADS=4
export PADDLE_OCR_ENABLE_MKLDNN=true

# Model settings
export PADDLE_OCR_DEFAULT_LANG=ch
export PADDLE_OCR_MODEL_VERSION=PP-OCRv5
```

### Configuration File (config.yaml)
```yaml
service:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: "INFO"
  timeout: 120

performance:
  enable_gpu: false
  enable_mkldnn: true
  cpu_threads: 4
  batch_size: 1
  memory_optimization: true

security:
  allowed_origins: ["https://your-domain.com"]
  rate_limiting:
    enabled: true
    requests_per_minute: 60
```

### Load Balancing Configuration
```bash
# Run multiple workers
for i in {8001..8003}; do
  nohup python3 paddleocr_service.py serve --port $i > logs/worker_$i.log 2>&1 &
done

# Update nginx upstream
# Add servers to upstream block in nginx config
```

## Monitoring

### Health Checks
```bash
# Basic health check
curl -f http://localhost:8000/health

# Detailed status
curl http://localhost:8000/ | jq

# Performance test
python3 examples/test_ocr.py --test performance --iterations 10
```

### Logging
```bash
# View logs
tail -f logs/service.log

# Systemd logs
sudo journalctl -u paddleocr -f

# Docker logs
docker-compose logs -f paddleocr-service
```

### Metrics Collection
```bash
# Install monitoring tools
pip install prometheus-client grafana-api

# Configure metrics endpoint
# Add to service configuration
```

### Alerting
```bash
# Create monitoring script
#!/bin/bash
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "PaddleOCR service is down!" | mail -s "Service Alert" admin@company.com
fi
```

## Performance Optimization

### CPU Optimization
```bash
# Set CPU affinity
taskset -c 0-3 python3 paddleocr_service.py serve

# Enable MKL-DNN
export PADDLE_OCR_ENABLE_MKLDNN=true
export OMP_NUM_THREADS=4
```

### Memory Optimization
```bash
# Configure memory limits
ulimit -v 4194304  # 4GB virtual memory limit

# Use memory-efficient models
export PADDLE_OCR_MODEL_VERSION=PP-OCRv5_mobile
```

### GPU Acceleration
```bash
# Install CUDA toolkit
# Configure GPU support
export PADDLE_OCR_ENABLE_GPU=true
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check Python dependencies
python3 -c "import paddleocr; print('OK')"

# Check port availability
sudo netstat -tlnp | grep :8000

# Check permissions
ls -la paddleocr_service.py
```

#### Model Download Issues
```bash
# Manual model download
python3 -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR()
"

# Check network connectivity
curl -I https://paddleocr.bj.bcebos.com/

# Use local models
export PADDLE_OCR_MODELS_DIR=/path/to/local/models
```

#### Memory Issues
```bash
# Monitor memory usage
free -h
top -p $(pgrep -f paddleocr)

# Use mobile models
export PADDLE_OCR_MODEL_VERSION=PP-OCRv5_mobile

# Reduce batch size
export PADDLE_OCR_BATCH_SIZE=1
```

#### Performance Issues
```bash
# Check CPU usage
htop

# Enable performance optimizations
export PADDLE_OCR_ENABLE_MKLDNN=true
export OMP_NUM_THREADS=4

# Use multiple workers
python3 paddleocr_service.py serve --workers 4
```

### Log Analysis
```bash
# Search for errors
grep -i error logs/service.log

# Monitor request times
grep "processing time" logs/service.log | tail -20

# Check model loading
grep "model" logs/service.log
```

### Debug Mode
```bash
# Start in debug mode
python3 paddleocr_service.py serve --log-level DEBUG

# Enable verbose logging
export PADDLE_OCR_LOG_LEVEL=DEBUG
```

## Security Considerations

### Network Security
- Use HTTPS in production
- Configure firewall rules
- Implement rate limiting
- Use VPN for internal access

### Application Security
- Validate all inputs
- Sanitize file uploads
- Implement authentication if needed
- Regular security updates

### Data Security
- No persistent image storage
- Secure API key management
- Audit logs
- Data encryption in transit

## Backup and Recovery

### Configuration Backup
```bash
# Backup configuration
tar -czf backup-config-$(date +%Y%m%d).tar.gz \
  .env config.yaml docker-compose.yml

# Backup models
tar -czf backup-models-$(date +%Y%m%d).tar.gz models/
```

### Disaster Recovery
```bash
# Restore configuration
tar -xzf backup-config-YYYYMMDD.tar.gz

# Restart services
sudo systemctl restart paddleocr
sudo systemctl restart nginx
```

## Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple instances
- Container orchestration (Kubernetes)
- Auto-scaling based on load

### Vertical Scaling
- Increase CPU/memory resources
- Use GPU acceleration
- Optimize model configurations
- Tune performance parameters

---

For additional support, please refer to the main README.md or create an issue in the GitHub repository.