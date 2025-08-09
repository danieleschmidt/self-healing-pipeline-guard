# Healing Guard - Production Deployment Guide

## 🚀 System Overview

Healing Guard is a production-ready, AI-powered self-healing CI/CD pipeline monitoring system with quantum-inspired optimization capabilities. This guide provides comprehensive instructions for deploying the system in production environments.

## ✨ System Capabilities

### Core Features
- **Self-Healing CI/CD Pipeline Monitoring**: Automatically detect, diagnose, and fix pipeline failures
- **Quantum-Inspired Task Optimization**: Advanced scheduling algorithms for optimal resource utilization
- **AI-Powered Failure Detection**: Machine learning-based pattern recognition for failure classification
- **Automated Healing Strategy Generation**: Intelligent strategy selection and execution
- **Advanced Caching & Performance Optimization**: Multi-level caching with LRU eviction
- **Comprehensive Observability**: Distributed tracing, metrics, and monitoring
- **Security-Hardened Input Validation**: Protection against injection attacks
- **Adaptive Load Balancing & Auto-Scaling**: Intelligent resource management

### Architecture Generations
- **Generation 1**: Core API functionality (✅ Complete)
- **Generation 2**: Robust reliability features (✅ Complete)
- **Generation 3**: Advanced scaling & optimization (✅ Complete)

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 8GB RAM, Recommended 16GB+ RAM
- **Storage**: Minimum 20GB available space
- **Network**: High-bandwidth connection for CI/CD monitoring

### Dependencies

#### Required Python Packages
```bash
# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Optional (recommended for full functionality)
redis>=5.0.0
asyncpg>=0.29.0
prometheus-client>=0.19.0
psutil>=5.9.0
```

#### System Packages
```bash
# Ubuntu/Debian
apt update && apt install -y \
    python3-numpy python3-fastapi python3-uvicorn \
    python3-pydantic python3-scipy python3-sklearn \
    python3-psutil python3-prometheus-client

# Optional: Database and cache
apt install -y postgresql redis-server
```

## 🏗️ Installation Steps

### 1. Clone and Setup
```bash
git clone <repository-url>
cd healing-guard
```

### 2. Environment Configuration
Create `.env` file:
```bash
# Application Settings
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-secure-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Database (Optional)
DATABASE_URL=postgresql://user:password@localhost:5432/healing_guard

# Redis (Optional)
REDIS_URL=redis://localhost:6379/0

# Monitoring
METRICS_ENABLED=true
LOG_LEVEL=INFO
PROMETHEUS_PORT=9090

# External Integrations
GITHUB_APP_ID=your_github_app_id
GITHUB_PRIVATE_KEY_PATH=/path/to/private-key.pem
GITLAB_TOKEN=your_gitlab_token
SLACK_WEBHOOK_URL=your_slack_webhook
```

### 3. Install Dependencies
```bash
# Option A: Using apt (recommended for Ubuntu/Debian)
sudo apt install -y python3-numpy python3-fastapi python3-uvicorn \
    python3-pydantic python3-scipy python3-sklearn python3-psutil

# Option B: Using pip with virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Database Setup (Optional)
```bash
# PostgreSQL setup
sudo -u postgres psql
CREATE DATABASE healing_guard;
CREATE USER healing_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE healing_guard TO healing_user;
```

### 5. Redis Setup (Optional)
```bash
# Install and start Redis
sudo apt install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

## 🚀 Deployment Options

### Option 1: Direct Python Deployment
```bash
# Start the server
python3 server.py

# Or with custom settings
HOST=0.0.0.0 PORT=8000 python3 server.py
```

### Option 2: Using Systemd Service
Create `/etc/systemd/system/healing-guard.service`:
```ini
[Unit]
Description=Healing Guard API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/healing-guard
Environment=ENVIRONMENT=production
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable healing-guard
sudo systemctl start healing-guard
```

### Option 3: Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-numpy python3-scipy python3-sklearn \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . .

# Install Python dependencies
RUN pip install fastapi uvicorn pydantic psutil

# Expose port
EXPOSE 8000

# Start server
CMD ["python3", "server.py"]
```

Build and run:
```bash
docker build -t healing-guard .
docker run -d -p 8000:8000 --name healing-guard healing-guard
```

### Option 4: Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healing-guard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: healing-guard
  template:
    metadata:
      labels:
        app: healing-guard
    spec:
      containers:
      - name: healing-guard
        image: healing-guard:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: healing-guard-service
spec:
  selector:
    app: healing-guard
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔧 Configuration

### Performance Tuning
```bash
# Environment variables for optimization
export QUANTUM_MAX_PARALLEL_TASKS=8
export QUANTUM_OPTIMIZATION_ITERATIONS=1000
export HEALING_MAX_CONCURRENT=5
export DB_POOL_SIZE=20
export REDIS_MAX_CONNECTIONS=100
```

### Security Configuration
```bash
# Generate secure secret key
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Set restrictive CORS origins
export CORS_ORIGINS="https://your-domain.com,https://ci.your-domain.com"
```

### Monitoring Setup
```bash
# Enable comprehensive metrics
export METRICS_ENABLED=true
export HEALTH_CHECK_INTERVAL=30

# Optional: Jaeger tracing
export JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

## 📊 Health Checks & Monitoring

### Health Endpoints
- **Liveness**: `GET /health/live` - Basic service health
- **Readiness**: `GET /health/ready` - Service ready for traffic
- **Comprehensive**: `GET /health` - Detailed health status

### Metrics Endpoints
- **Prometheus**: `GET /metrics` - Prometheus-compatible metrics
- **System Status**: `GET /api/v1/system/status` - System overview
- **System Metrics**: `GET /api/v1/system/metrics` - Detailed metrics

### Monitoring Integration
```bash
# Prometheus scrape config
- job_name: 'healing-guard'
  static_configs:
    - targets: ['localhost:8000']
  metrics_path: /metrics
  scrape_interval: 30s
```

## 🛡️ Security Considerations

### Network Security
- Use HTTPS in production (configure reverse proxy)
- Implement proper firewall rules
- Restrict access to health/metrics endpoints

### Authentication & Authorization
- Configure JWT with strong secret keys
- Implement proper RBAC for API endpoints
- Use service accounts for CI/CD integrations

### Input Validation
- All inputs are automatically validated and sanitized
- SQL injection protection enabled
- Shell injection protection enabled
- Path traversal protection enabled

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install missing dependencies
sudo apt install python3-numpy python3-scipy python3-sklearn
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U healing_user -d healing_guard
```

#### 3. Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping
```

#### 4. Performance Issues
```bash
# Monitor system resources
htop
iotop

# Check application logs
journalctl -u healing-guard -f

# Review metrics endpoint
curl http://localhost:8000/metrics
```

### Log Analysis
```bash
# Application logs
tail -f /var/log/healing-guard/app.log

# System logs
journalctl -u healing-guard -f

# Performance logs
grep "slow" /var/log/healing-guard/app.log
```

## 📈 Scaling & High Availability

### Horizontal Scaling
```bash
# Multiple instances behind load balancer
# Instance 1
PORT=8001 python3 server.py &

# Instance 2  
PORT=8002 python3 server.py &

# Instance 3
PORT=8003 python3 server.py &
```

### Load Balancer Configuration (Nginx)
```nginx
upstream healing_guard {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://healing_guard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /health {
        proxy_pass http://healing_guard;
        access_log off;
    }
}
```

### Database High Availability
- Use PostgreSQL streaming replication
- Implement connection pooling
- Configure automatic failover

### Redis High Availability
- Use Redis Sentinel for automatic failover
- Implement Redis Cluster for scaling
- Configure appropriate persistence settings

## 🔄 Backup & Recovery

### Database Backup
```bash
# Daily backup
pg_dump healing_guard > backup_$(date +%Y%m%d).sql

# Automated backup script
0 2 * * * /path/to/backup_script.sh
```

### Configuration Backup
```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
    .env healing_guard/ logs/
```

### Recovery Procedures
```bash
# Database recovery
psql healing_guard < backup_20241201.sql

# Configuration recovery
tar -xzf config_backup_20241201.tar.gz
```

## 📝 API Documentation

Once deployed, comprehensive API documentation is available at:
- **Interactive Docs**: `http://your-domain:8000/docs`
- **ReDoc**: `http://your-domain:8000/redoc`
- **OpenAPI JSON**: `http://your-domain:8000/openapi.json`

## 🎯 Production Checklist

### Pre-Deployment
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] Database setup completed
- [ ] SSL certificates configured
- [ ] Firewall rules implemented
- [ ] Monitoring setup verified

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Log aggregation configured
- [ ] Backup procedures tested
- [ ] Performance benchmarks established
- [ ] Security scan completed

### Ongoing Maintenance
- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Log rotation configured
- [ ] Backup verification
- [ ] Capacity planning
- [ ] Documentation updates

## 📞 Support & Maintenance

### Performance Monitoring
- Monitor response times (target: <200ms)
- Track error rates (target: <1%)
- Monitor resource utilization
- Set up alerting for anomalies

### Regular Maintenance Tasks
- Update dependencies monthly
- Review security logs weekly
- Performance optimization quarterly
- Capacity planning reviews

### Emergency Procedures
1. **Service Down**: Check systemd status, review logs
2. **High CPU**: Review optimization metrics, scale horizontally
3. **Memory Issues**: Check for memory leaks, restart service
4. **Database Issues**: Check connection pool, verify backup integrity

## 🚀 Next Steps

After successful deployment:
1. **Integration**: Connect to your CI/CD platforms
2. **Training**: Train team on system capabilities
3. **Optimization**: Fine-tune performance based on usage patterns
4. **Monitoring**: Set up comprehensive alerting and dashboards
5. **Expansion**: Plan for additional features and scaling

---

## 🎉 Congratulations!

Your Healing Guard system is now deployed and ready to revolutionize your CI/CD pipeline reliability. The system will automatically:

- Monitor your pipelines 24/7
- Detect failures using AI-powered analysis
- Generate and execute healing strategies
- Optimize resource utilization with quantum-inspired algorithms
- Scale automatically based on demand
- Provide comprehensive observability and metrics

For additional support or advanced configuration options, please refer to the comprehensive API documentation available at your deployment endpoint.

**Welcome to the future of self-healing CI/CD pipelines!** 🚀✨