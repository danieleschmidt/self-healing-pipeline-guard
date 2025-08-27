# Healing Guard Implementation Guide

## Quick Start

This guide will help you deploy and configure Healing Guard in your environment.

## Prerequisites

### System Requirements
- **Operating System:** Linux (Ubuntu 20.04+ recommended)
- **Memory:** 8GB RAM minimum, 16GB recommended
- **CPU:** 4 cores minimum, 8 cores recommended
- **Storage:** 50GB available space minimum
- **Network:** Internet connectivity for external integrations

### Software Dependencies
- **Docker:** 20.10+
- **Docker Compose:** 2.0+
- **Kubernetes:** 1.24+ (for production deployment)
- **Python:** 3.9+ (for development)

## Installation Options

### Option 1: Docker Compose (Recommended for Testing)

1. **Clone the repository:**
```bash
git clone https://github.com/terragon-labs/healing-guard.git
cd healing-guard
```

2. **Start the services:**
```bash
docker-compose up -d
```

3. **Verify installation:**
```bash
curl http://localhost:8000/api/health
```

### Option 2: Production Kubernetes Deployment

1. **Prepare environment:**
```bash
# Set environment variables
export NAMESPACE=healing-guard
export IMAGE_TAG=latest
export REGISTRY=ghcr.io/terragon-labs
```

2. **Run deployment script:**
```bash
chmod +x scripts/production-deploy.sh
./scripts/production-deploy.sh
```

3. **Monitor deployment:**
```bash
kubectl get pods -n healing-guard
kubectl logs -f deployment/healing-guard-api -n healing-guard
```

### Option 3: Manual Python Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Initialize database:**
```bash
python -m healing_guard.core.database migrate
```

4. **Start the service:**
```bash
python -m healing_guard.api.main
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/healing_guard
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET=your-super-secret-jwt-key
ENCRYPTION_KEY=your-32-byte-encryption-key

# Monitoring
PROMETHEUS_ENDPOINT=http://localhost:9090
GRAFANA_ENDPOINT=http://localhost:3000

# Notification Channels
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SMTP_HOST=smtp.gmail.com
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Multi-tenant Configuration (Optional)
MULTI_TENANT_MODE=true
DEFAULT_TENANT_QUOTA=100

# AI/ML Configuration
ML_MODEL_PATH=/opt/healing-guard/models
PREDICTION_WINDOW_HOURS=24
```

### Webhook Configuration

Configure your CI/CD system to send webhooks to Healing Guard:

**GitHub Actions Example:**
```yaml
name: Notify Healing Guard
on:
  workflow_run:
    workflows: ["CI Pipeline"]
    types: [completed]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Send webhook
        run: |
          curl -X POST ${{ secrets.HEALING_GUARD_WEBHOOK }} \
            -H "Content-Type: application/json" \
            -d '{
              "pipeline_id": "${{ github.workflow }}",
              "status": "${{ github.workflow_run.conclusion }}",
              "commit": "${{ github.sha }}",
              "branch": "${{ github.ref_name }}"
            }'
```

**Jenkins Pipeline Example:**
```groovy
pipeline {
    agent any
    post {
        always {
            script {
                def payload = [
                    pipeline_id: env.JOB_NAME,
                    status: currentBuild.result,
                    build_number: env.BUILD_NUMBER,
                    commit: env.GIT_COMMIT
                ]
                
                httpRequest(
                    url: "${HEALING_GUARD_WEBHOOK}",
                    httpMode: 'POST',
                    contentType: 'APPLICATION_JSON',
                    requestBody: groovy.json.JsonOutput.toJson(payload)
                )
            }
        }
    }
}
```

## Initial Setup

### 1. Register Your First Pipeline

```bash
curl -X POST http://localhost:8000/api/pipelines \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "name": "Production Deployment",
    "webhook_url": "https://your-ci-system.com/webhook",
    "config": {
      "retry_attempts": 3,
      "timeout": 300,
      "notification_channels": ["slack", "email"]
    }
  }'
```

### 2. Configure Authentication

Create an admin user:

```bash
python -m healing_guard.cli create-user \
  --email admin@yourcompany.com \
  --password secure_password \
  --role admin
```

### 3. Set Up Monitoring

Import Grafana dashboards:

```bash
# Copy dashboard configurations
cp grafana/dashboards/* /var/lib/grafana/dashboards/

# Restart Grafana
docker-compose restart grafana
```

## Usage Examples

### Basic Pipeline Monitoring

1. **Register a pipeline:**
```python
from healing_guard import HealingGuardClient

client = HealingGuardClient("http://localhost:8000", "your-jwt-token")

pipeline = client.pipelines.create({
    "name": "My Pipeline",
    "webhook_url": "https://ci.example.com/webhook"
})
```

2. **Monitor pipeline health:**
```python
# Get pipeline status
status = client.pipelines.get_status(pipeline.id)
print(f"Pipeline status: {status.health}")

# Get recent failures
failures = client.failures.list(pipeline_id=pipeline.id, limit=10)
for failure in failures:
    print(f"Failure: {failure.type} at {failure.detected_at}")
```

### Advanced ML-Based Monitoring

1. **Enable pattern recognition:**
```python
# Configure ML monitoring
client.ml.configure_pattern_recognition(
    pipeline_id=pipeline.id,
    models=["random_forest", "isolation_forest"],
    sensitivity=0.8
)
```

2. **Get predictions:**
```python
# Request failure prediction
prediction = client.ml.predict_failure(
    pipeline_id=pipeline.id,
    window_hours=24
)

if prediction.probability > 0.7:
    print(f"High failure risk: {prediction.most_likely_type}")
```

### Enterprise Security Setup

1. **Configure multi-tenancy:**
```python
# Create tenant
tenant = client.tenants.create({
    "name": "Acme Corporation",
    "admin_email": "admin@acme.com",
    "resource_quota": {
        "max_pipelines": 100,
        "max_storage_gb": 500
    }
})
```

2. **Set up compliance auditing:**
```python
# Configure compliance monitoring
client.compliance.configure({
    "standards": ["sox", "gdpr"],
    "audit_frequency": "daily",
    "notification_email": "compliance@yourcompany.com"
})
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```
Error: Could not connect to database
```
**Solution:**
- Verify PostgreSQL is running: `docker-compose ps postgres`
- Check connection string in `.env` file
- Ensure database exists: `createdb healing_guard`

#### 2. Redis Connection Error
```
Error: Redis connection refused
```
**Solution:**
- Start Redis: `docker-compose up redis -d`
- Verify Redis URL in configuration
- Check Redis logs: `docker-compose logs redis`

#### 3. Webhook Not Receiving Data
```
Warning: No webhook data received
```
**Solution:**
- Verify webhook URL is accessible from CI system
- Check firewall rules and network configuration
- Test webhook endpoint: `curl -X POST your-webhook-url`

#### 4. ML Models Not Loading
```
Error: Could not load ML models
```
**Solution:**
- Check model file permissions: `ls -la /opt/healing-guard/models/`
- Verify Python dependencies: `pip install scikit-learn pandas`
- Download models: `python -m healing_guard.ml.download_models`

#### 5. High Memory Usage
```
Warning: Memory usage above 80%
```
**Solution:**
- Increase container memory limits
- Optimize model loading: Set `ML_LAZY_LOADING=true`
- Enable data compression: Set `COMPRESS_METRICS=true`

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python -m healing_guard.api.main
```

View debug logs:

```bash
# Docker Compose
docker-compose logs -f api

# Kubernetes
kubectl logs -f deployment/healing-guard-api -n healing-guard
```

### Performance Tuning

#### Database Optimization

```sql
-- Add indexes for better performance
CREATE INDEX idx_failures_pipeline_id ON failures(pipeline_id);
CREATE INDEX idx_failures_detected_at ON failures(detected_at);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
```

#### Redis Configuration

```bash
# Add to redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### API Performance

```bash
# Increase worker processes
export WORKERS=4
export WORKER_CONNECTIONS=1000

# Enable caching
export ENABLE_CACHE=true
export CACHE_TTL=300
```

## Monitoring and Maintenance

### Health Checks

Set up automated health checks:

```bash
#!/bin/bash
# health-check.sh

API_ENDPOINT="http://localhost:8000/api/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_ENDPOINT)

if [ $RESPONSE -eq 200 ]; then
    echo "API is healthy"
    exit 0
else
    echo "API health check failed with code: $RESPONSE"
    exit 1
fi
```

### Log Rotation

Configure log rotation:

```bash
# /etc/logrotate.d/healing-guard
/var/log/healing-guard/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 healing-guard healing-guard
}
```

### Backup Strategy

#### Database Backup
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/opt/healing-guard/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

pg_dump $DATABASE_URL | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete
```

#### Configuration Backup
```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
    .env \
    docker-compose.yml \
    k8s/ \
    grafana/dashboards/
```

### Monitoring Alerts

Set up Prometheus alerts:

```yaml
# alerts.yml
groups:
  - name: healing-guard
    rules:
      - alert: HighFailureRate
        expr: rate(healing_guard_failures_total[5m]) > 0.1
        for: 2m
        annotations:
          summary: "High failure rate detected"
          
      - alert: APIResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "API response time is high"
```

## Upgrading

### Version Upgrade Process

1. **Backup current installation:**
```bash
./scripts/backup.sh
```

2. **Stop services:**
```bash
docker-compose down
```

3. **Update to new version:**
```bash
git pull origin main
docker-compose pull
```

4. **Run database migrations:**
```bash
docker-compose run api python -m healing_guard.core.database migrate
```

5. **Start services:**
```bash
docker-compose up -d
```

6. **Verify upgrade:**
```bash
curl http://localhost:8000/api/health
```

### Rolling Updates (Kubernetes)

```bash
# Update image tag
kubectl set image deployment/healing-guard-api \
    api=ghcr.io/terragon-labs/healing-guard:v4.1.0 \
    -n healing-guard

# Monitor rollout
kubectl rollout status deployment/healing-guard-api -n healing-guard

# Rollback if needed
kubectl rollout undo deployment/healing-guard-api -n healing-guard
```

## Support

For implementation support:

- **Documentation:** https://docs.terragon.ai/healing-guard
- **Community Forum:** https://community.terragon.ai
- **Enterprise Support:** support@terragon.ai
- **Emergency Hotline:** +1-800-TERRAGON

---
*Implementation Guide v4.0 - Terragon Labs*