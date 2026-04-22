# Deployment Guide

## AI-Based Network Attack Detection System

This guide provides comprehensive instructions for deploying the AI-based network attack detection system in production environments.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Options](#deployment-options)
4. [Production Setup](#production-setup)
5. [Configuration for Production](#configuration-for-production)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Scaling Strategies](#scaling-strategies)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores @ 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+, Windows Server 2019+
- **Python**: 3.8 or higher
- **Network**: 1 Gbps connection

### Recommended Requirements

- **CPU**: 8+ cores @ 3.0 GHz
- **RAM**: 16 GB+
- **Storage**: 100 GB NVMe SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for neural networks)
- **Network**: 10 Gbps connection
- **Redundancy**: Multiple servers with load balancing

---

## Pre-Deployment Checklist

### Data Preparation

- [ ] Download and prepare training datasets (CICIDS2017, UNSW-NB15)
- [ ] Verify data quality and completeness
- [ ] Set up data storage infrastructure
- [ ] Configure backup solutions

### Model Training

- [ ] Train models on representative data
- [ ] Validate model performance (>95% accuracy target)
- [ ] Test on holdout dataset
- [ ] Save trained models
- [ ] Document model versions

### Infrastructure

- [ ] Set up production server(s)
- [ ] Configure network access
- [ ] Install dependencies
- [ ] Set up logging infrastructure
- [ ] Configure monitoring tools
- [ ] Implement backup systems

### Security

- [ ] Review security policies
- [ ] Set up access controls
- [ ] Configure firewall rules
- [ ] Enable encryption (data at rest and in transit)
- [ ] Implement authentication mechanisms

---

## Deployment Options

### Option 1: Standalone Server

**Best for**: Small to medium networks, pilot deployments

```bash

# Clone repository
git clone https://github.com/yourusername/network-attack-detection.git
cd network-attack-detection

# Run setup
./setup.sh

# Train models
cd src
python train.py

# Start real-time detection
python realtime_detection.py
```

**Pros**:
- Simple setup
- Easy to manage
- Lower cost

**Cons**:
- Single point of failure
- Limited scalability
- Manual updates required

### Option 2: Docker Container

**Best for**: Consistent environments, easy deployment

```bash

# Build Docker image
docker build -t network-attack-detection:latest .

# Run container
docker run -d \
  --name attack-detector \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -p 8080:8080 \
  network-attack-detection:latest
```

**Pros**:
- Consistent environment
- Easy updates
- Portable
- Isolated

**Cons**:
- Overhead
- Requires Docker knowledge
- Storage management needed

### Option 3: Kubernetes Cluster

**Best for**: Large-scale deployments, enterprise environments

```bash

# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Scale deployment
kubectl scale deployment attack-detector --replicas=5
```

**Pros**:
- High availability
- Auto-scaling
- Self-healing
- Load balancing

**Cons**:
- Complex setup
- Requires Kubernetes expertise
- Higher operational overhead

### Option 4: Cloud Deployment

**Best for**: Cloud-native applications, variable workloads

#### AWS

```bash


# Deploy to AWS EC2
aws ec2 run-instances \
  --image-id ami-xxxxxxxx \
  --instance-type c5.2xlarge \
  --key-name your-key \
  --security-groups attack-detection-sg

# Or use AWS Lambda for serverless
aws lambda create-function \
  --function-name attack-detector \
  --runtime python3.8 \
  --handler lambda_function.handler \
  --zip-file fileb://function.zip
```

#### Google Cloud

```bash

# Deploy to GCP Compute Engine
gcloud compute instances create attack-detector \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud
```

#### Azure

```bash

# Deploy to Azure VM
az vm create \
  --resource-group attack-detection-rg \
  --name attack-detector \
  --image UbuntuLTS \
  --size Standard_D4s_v3
```

---

## Production Setup

### Step 1: Server Configuration

```bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.8 python3-pip nginx supervisor

# Create deployment user
sudo useradd -m -s /bin/bash detector
sudo usermod -aG sudo detector
```

### Step 2: Application Setup

```bash


# Switch to deployment user
sudo su - detector

# Clone repository
git clone https://github.com/yourusername/network-attack-detection.git
cd network-attack-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set permissions
chmod +x setup.sh
chmod -R 755 src/
```

### Step 3: Model Deployment

```bash


# Copy pre-trained models
cp /path/to/trained/models/* data/models/

# Verify models
python -c "from src.models.ml_models import RandomForestModel; \
           m = RandomForestModel(); \
           m.load('data/models'); \
           print('✓ Models loaded successfully')"
```

### Step 4: Process Management with Supervisor

Create `/etc/supervisor/conf.d/attack-detector.conf`:

```ini

[program:attack-detector]
command=/home/detector/network-attack-detection/venv/bin/python src/realtime_detection.py
directory=/home/detector/network-attack-detection
user=detector
autostart=true
autorestart=true
stderr_logfile=/var/log/attack-detector/error.log
stdout_logfile=/var/log/attack-detector/output.log
environment=PYTHONPATH="/home/detector/network-attack-detection/src"

```

Start the service:

```bash

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start attack-detector
sudo supervisorctl status
```

### Step 5: Nginx Reverse Proxy (Optional)

Create `/etc/nginx/sites-available/attack-detector`:

```nginx
server {
    listen 80;
    server_name attack-detector.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable and start:

```bash

sudo ln -s /etc/nginx/sites-available/attack-detector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Configuration for Production

### 1. Edit `configs/config.yaml`

```yaml
# Production configuration
project:
  name: "Network Attack Detection - PRODUCTION"
  version: "1.0.0"

# Optimize for production
preprocessing:
  handle_missing: "drop"
  outlier_detection: true
  balance_classes: false  # Already balanced during training

# Enable real-time detection
realtime:
  enabled: true
  buffer_size: 10000
  prediction_threshold: 0.8
  alert_threshold: 0.95
  
  alerts:
    enable_logging: true
    enable_email: true
    enable_webhook: true
    webhook_url: "https://your-webhook-url.com/alert"

# Logging
logging:
  level: "INFO"
  file: "/var/log/attack-detector/system.log"
  console: false
  max_bytes: 52428800  # 50MB
  backup_count: 10

# Resources
resources:
  max_workers: 8
  memory_limit: "12GB"
  gpu_enabled: true
```

### 2. Environment Variables

Create `.env` file:

```bash


# Production environment variables
ENVIRONMENT=production
LOG_LEVEL=INFO
MODEL_PATH=/home/detector/network-attack-detection/data/models
DATA_PATH=/home/detector/network-attack-detection/data

# Database (if using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=attack_detection
DB_USER=detector
DB_PASSWORD=your_secure_password

# Email alerts
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@example.com
SMTP_PASSWORD=your_email_password
ALERT_RECIPIENTS=security-team@example.com

# Webhook alerts
WEBHOOK_ENABLED=true
WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 3. Systemd Service (Alternative to Supervisor)

Create `/etc/systemd/system/attack-detector.service`:

```ini
[Unit]
Description=Network Attack Detection Service
After=network.target

[Service]
Type=simple
User=detector
WorkingDirectory=/home/detector/network-attack-detection
Environment="PATH=/home/detector/network-attack-detection/venv/bin"
EnvironmentFile=/home/detector/network-attack-detection/.env
ExecStart=/home/detector/network-attack-detection/venv/bin/python src/realtime_detection.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash


sudo systemctl daemon-reload
sudo systemctl enable attack-detector
sudo systemctl start attack-detector
sudo systemctl status attack-detector
```

---

## Monitoring and Maintenance

### Log Monitoring

```bash


# System logs
tail -f /var/log/attack-detector/system.log

# Security alerts
tail -f /var/log/attack-detector/alerts.log

# Performance metrics
tail -f /var/log/attack-detector/performance.log
```

### Health Checks

Create monitoring script `/home/detector/health_check.sh`:

```bash


#!/bin/bash

# Check if service is running
if ! systemctl is-active --quiet attack-detector; then
    echo "ERROR: Service is not running"
    systemctl restart attack-detector
    exit 1
fi

# Check log for errors
if grep -q "ERROR" /var/log/attack-detector/system.log | tail -100; then
    echo "WARNING: Errors found in logs"
fi

# Check disk space
USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$USAGE" -gt 80 ]; then
    echo "WARNING: Disk usage is at ${USAGE}%"
fi

echo "✓ Health check passed"
```

Schedule with cron:

```bash


# Run every 5 minutes
*/5 * * * * /home/detector/health_check.sh >> /var/log/health_check.log 2>&1
```

### Model Retraining

Schedule periodic retraining:

```bash


# Retrain weekly on Sunday at 2 AM
0 2 * * 0 cd /home/detector/network-attack-detection/src && /home/detector/network-attack-detection/venv/bin/python train.py >> /var/log/attack-detector/training.log 2>&1
```

### Performance Monitoring

Use Prometheus and Grafana:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'attack-detector'
    static_configs:
      - targets: ['localhost:9090']
```

---

## Scaling Strategies

### Horizontal Scaling

```bash


# Add more detection servers
for i in {1..5}; do
  ./deploy_server.sh detector-node-$i
done

# Configure load balancer
# Update nginx upstream configuration
```

### Vertical Scaling

```bash


# Upgrade server resources
# Increase CPU cores
# Add more RAM
# Use faster storage (NVMe SSD)
```

### Database Optimization

```sql
-- Add indexes for faster queries
CREATE INDEX idx_timestamp ON detections(timestamp);
CREATE INDEX idx_attack_type ON detections(attack_type);
CREATE INDEX idx_confidence ON detections(confidence);
```

---

## Security Considerations

### 1. Access Control

```bash


# Restrict file permissions
chmod 700 /home/detector/network-attack-detection
chmod 600 /home/detector/network-attack-detection/.env
chmod 600 /home/detector/network-attack-detection/configs/config.yaml
```

### 2. Firewall Configuration

```bash


# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 3. SSL/TLS Configuration

```bash

# Install Let's Encrypt certificate
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d attack-detector.example.com
```

### 4. Regular Updates

```bash

# Update dependencies
pip install --upgrade -r requirements.txt

# Update system packages
sudo apt update && sudo apt upgrade
```

---

## Troubleshooting

### Service Won't Start

```bash


# Check logs
sudo journalctl -u attack-detector -n 50

# Check Python path
which python3
python3 --version

# Verify dependencies
pip list | grep -E 'tensorflow|scikit-learn|xgboost'
```

### High Memory Usage

```bash

# Check memory usage
free -h
ps aux --sort=-%mem | head

# Reduce batch size in config
# Limit number of workers
```

### Poor Detection Performance

```bash


# Retrain models with more data
python src/train.py

# Verify model files
ls -lh data/models/

# Check prediction threshold
# Adjust in configs/config.yaml
```

### Log Files Growing Too Large

```bash


# Configure log rotation
sudo vim /etc/logrotate.d/attack-detector

# Add:
/var/log/attack-detector/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 detector detector
}
```

---

## Support

For deployment issues:
- Check logs: `/var/log/attack-detector/`
- Review documentation: `docs/`
- Open issue on GitHub
- Contact: aljabid444@gmail.com

---

**Last Updated**: April 2026
