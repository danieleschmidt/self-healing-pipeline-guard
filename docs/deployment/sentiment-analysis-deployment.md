# Sentiment Analysis System Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Self-Healing Pipeline Guard's sentiment analysis system, which adds intelligent emotional context analysis to CI/CD pipeline failures for enhanced healing prioritization and developer experience.

## Architecture Overview

The sentiment analysis system consists of several interconnected components:

```mermaid
graph TB
    A[CI/CD Pipeline Failure] --> B[Failure Detector]
    B --> C[Sentiment Analyzer]
    C --> D[Healing Engine]
    D --> E[Prioritized Healing Actions]
    
    C --> F[Compliance Layer]
    F --> G[GDPR/CCPA/PDPA Processing]
    
    C --> H[Caching Layer]
    C --> I[Metrics Collection]
    C --> J[Auto-scaling]
    
    K[API Gateway] --> L[Sentiment Routes]
    L --> C
    
    M[Internationalization] --> N[8 Languages]
    N --> C
```

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Python**: 3.11+
- **Dependencies**: See `pyproject.toml`

### Recommended Production Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **Load Balancer**: For multi-instance deployment
- **Database**: PostgreSQL 13+ or Redis 6+
- **Monitoring**: Prometheus + Grafana

## Pre-deployment Checklist

### 1. Environment Preparation
- [ ] Python 3.11+ installed
- [ ] Poetry package manager installed
- [ ] Docker and Docker Compose available
- [ ] Access to required external APIs (if any)
- [ ] SSL certificates for HTTPS endpoints

### 2. Configuration Files
- [ ] Environment variables configured
- [ ] Database connection strings
- [ ] API keys and secrets in secure storage
- [ ] Logging configuration
- [ ] Monitoring endpoints configured

### 3. Security Setup
- [ ] HTTPS/TLS configured
- [ ] API rate limiting enabled
- [ ] Input validation activated
- [ ] Security headers configured
- [ ] CORS policies defined

## Deployment Methods

### Method 1: Docker Deployment (Recommended)

#### Step 1: Build the Container
```bash
# Clone the repository
git clone <repository-url>
cd self-healing-pipeline-guard

# Build the Docker image
docker build -t healing-guard-sentiment:latest .

# Verify the build
docker images | grep healing-guard-sentiment
```

#### Step 2: Configure Environment
Create a `.env` file:
```env
# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/healing_guard
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your-secret-key-here
API_KEY_VALIDATION=true
CORS_ORIGINS=https://yourdomain.com

# Compliance Configuration
GDPR_ENABLED=true
CCPA_ENABLED=true
PDPA_SINGAPORE_ENABLED=true
PDPA_THAILAND_ENABLED=true

# Monitoring Configuration
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_ENDPOINT=/health

# Internationalization
DEFAULT_LANGUAGE=en
SUPPORTED_LANGUAGES=en,es,fr,de,it,pt,ja,zh

# Performance Configuration
CACHE_TTL_SECONDS=1800
MAX_WORKER_THREADS=10
AUTO_SCALING_ENABLED=true
```

#### Step 3: Deploy with Docker Compose
Create `docker-compose.production.yml`:
```yaml
version: '3.8'

services:
  healing-guard-sentiment:
    image: healing-guard-sentiment:latest
    ports:
      - "8000:8000"
      - "9090:9090"  # Metrics endpoint
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: healing_guard
      POSTGRES_USER: healing_guard
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - healing-guard-sentiment
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Step 4: Start the Services
```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml logs healing-guard-sentiment
```

### Method 2: Kubernetes Deployment

#### Step 1: Create Kubernetes Manifests

Create `k8s/namespace.yaml`:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: healing-guard
  labels:
    name: healing-guard
```

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: healing-guard-sentiment
  namespace: healing-guard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: healing-guard-sentiment
  template:
    metadata:
      labels:
        app: healing-guard-sentiment
    spec:
      containers:
      - name: sentiment-analyzer
        image: healing-guard-sentiment:latest
        ports:
        - containerPort: 8000
          name: api
        - containerPort: 9090
          name: metrics
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: healing-guard-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: healing-guard-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 15
```

Create `k8s/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: healing-guard-sentiment-service
  namespace: healing-guard
spec:
  selector:
    app: healing-guard-sentiment
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

Create `k8s/ingress.yaml`:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: healing-guard-sentiment-ingress
  namespace: healing-guard
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.healing-guard.com
    secretName: healing-guard-tls
  rules:
  - host: api.healing-guard.com
    http:
      paths:
      - path: /sentiment
        pathType: Prefix
        backend:
          service:
            name: healing-guard-sentiment-service
            port:
              number: 8000
```

#### Step 2: Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -n healing-guard
kubectl get services -n healing-guard
kubectl get ingress -n healing-guard

# Check logs
kubectl logs -n healing-guard -l app=healing-guard-sentiment
```

### Method 3: Native Installation

#### Step 1: Install Dependencies
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone <repository-url>
cd self-healing-pipeline-guard
poetry install --with dev,test

# Activate virtual environment
poetry shell
```

#### Step 2: Configure Application
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

#### Step 3: Start Services
```bash
# Start the API server
uvicorn healing_guard.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# In another terminal, start the metrics server (if separate)
python -m healing_guard.metrics.server

# Start monitoring (if configured)
python -m healing_guard.monitoring.collector
```

## Configuration Options

### Core Sentiment Analysis Settings

```python
# In healing_guard/core/sentiment_analyzer.py
SENTIMENT_CONFIG = {
    "model_confidence_threshold": 0.7,
    "urgency_detection_threshold": 0.6,
    "frustration_keywords": ["fail", "error", "broken", "stuck"],
    "urgency_keywords": ["urgent", "critical", "production", "down"],
    "emotional_intensity_factors": {
        "production_environment": 1.5,
        "consecutive_failures": 1.3,
        "weekend_or_holiday": 1.2
    }
}
```

### Performance Configuration

```python
# Cache settings
CACHE_CONFIG = {
    "default_ttl": 1800,  # 30 minutes
    "max_memory_mb": 512,
    "eviction_policy": "lru",
    "adaptive_sizing": True
}

# Auto-scaling settings
SCALING_CONFIG = {
    "min_workers": 2,
    "max_workers": 20,
    "scale_up_threshold": 0.8,
    "scale_down_threshold": 0.3,
    "scaling_interval": 60
}
```

### Compliance Configuration

```python
# GDPR settings
GDPR_CONFIG = {
    "data_retention_days": 90,
    "consent_required": True,
    "anonymization_enabled": True,
    "audit_logging": True
}

# CCPA settings
CCPA_CONFIG = {
    "opt_out_enabled": True,
    "data_sale_tracking": False,
    "consumer_request_timeout_days": 45
}

# PDPA settings
PDPA_CONFIG = {
    "singapore_enabled": True,
    "thailand_enabled": True,
    "consent_duration_days": 365
}
```

## API Endpoints

### Core Endpoints

#### Analyze Single Text
```http
POST /sentiment/analyze
Content-Type: application/json
Authorization: Bearer <your-api-key>

{
  "text": "Build failed again, this is frustrating!",
  "context": {
    "repository": "my-app",
    "branch": "main",
    "is_production": true
  }
}
```

#### Analyze Pipeline Event
```http
POST /sentiment/analyze/pipeline-event
Content-Type: application/json
Authorization: Bearer <your-api-key>

{
  "event_type": "pipeline_failure",
  "message": "Tests are failing consistently",
  "metadata": {
    "repository": "my-app",
    "consecutive_failures": 3,
    "environment": "production"
  }
}
```

#### Batch Analysis
```http
POST /sentiment/analyze/batch
Content-Type: application/json
Authorization: Bearer <your-api-key>

{
  "texts": [
    "Build successful!",
    "Another test failure",
    "Deploy completed"
  ]
}
```

### Compliance Endpoints

#### GDPR Data Request
```http
POST /compliance/gdpr/request
Content-Type: application/json
Authorization: Bearer <your-api-key>

{
  "data_subject_id": "user123",
  "request_type": "access",
  "description": "Request access to personal data"
}
```

#### CCPA Opt-Out
```http
POST /compliance/ccpa/opt-out
Content-Type: application/json
Authorization: Bearer <your-api-key>

{
  "consumer_id": "user123",
  "opt_out_method": "web_form"
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Metrics
```http
GET /metrics
```

#### Performance Stats
```http
GET /sentiment/stats
Authorization: Bearer <your-api-key>
```

## Monitoring and Observability

### Prometheus Metrics

Key metrics to monitor:

```yaml
# Sentiment analysis metrics
sentiment_analysis_requests_total: Counter of total requests
sentiment_analysis_duration_seconds: Histogram of request durations
sentiment_analysis_cache_hits_total: Counter of cache hits
sentiment_analysis_cache_misses_total: Counter of cache misses

# Compliance metrics
compliance_requests_total: Counter by framework (GDPR/CCPA/PDPA)
compliance_violations_total: Counter of compliance violations
data_retention_cleanup_total: Counter of expired data cleanups

# Performance metrics
worker_pool_size: Current worker pool size
queue_depth: Current processing queue depth
error_rate: Rate of processing errors
```

### Grafana Dashboard

Import the provided dashboard configuration:

```json
{
  "dashboard": {
    "title": "Healing Guard Sentiment Analysis",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sentiment_analysis_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sentiment_analysis_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(sentiment_analysis_cache_hits_total[5m]) / (rate(sentiment_analysis_cache_hits_total[5m]) + rate(sentiment_analysis_cache_misses_total[5m])) * 100"
          }
        ]
      }
    ]
  }
}
```

### Log Configuration

Configure structured logging:

```python
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/sentiment_analysis.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "healing_guard.core.sentiment_analyzer": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}
```

## Security Configuration

### API Security

1. **Enable HTTPS**: Always use TLS 1.2+ in production
2. **API Authentication**: Implement Bearer token authentication
3. **Rate Limiting**: Configure per-endpoint rate limits
4. **Input Validation**: Enable comprehensive input sanitization
5. **CORS Configuration**: Restrict origins appropriately

```python
SECURITY_CONFIG = {
    "api_key_validation": True,
    "rate_limit_per_minute": 100,
    "max_request_size_mb": 10,
    "cors_origins": ["https://yourdomain.com"],
    "security_headers": {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block"
    }
}
```

### Data Security

1. **Encryption at Rest**: Configure database encryption
2. **Encryption in Transit**: Use TLS for all communications
3. **Data Anonymization**: Enable automatic PII anonymization
4. **Audit Logging**: Enable comprehensive audit trails

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
```bash
# Check memory usage
docker stats healing-guard-sentiment

# Adjust cache settings
# In .env file:
CACHE_MAX_MEMORY_MB=256
CACHE_EVICTION_POLICY=lru
```

#### 2. Slow Response Times
```bash
# Check worker pool status
curl http://localhost:8000/sentiment/stats

# Increase worker threads
# In .env file:
MAX_WORKER_THREADS=20
AUTO_SCALING_ENABLED=true
```

#### 3. Cache Issues
```bash
# Clear cache
curl -X POST http://localhost:8000/admin/cache/clear

# Check cache hit rate
curl http://localhost:8000/metrics | grep cache_hit_rate
```

#### 4. Compliance Errors
```bash
# Check compliance status
curl http://localhost:8000/compliance/status

# Run compliance audit
curl -X POST http://localhost:8000/admin/compliance/audit
```

### Log Analysis

#### Finding Performance Issues
```bash
# Search for slow requests
grep "sentiment_analysis.*duration.*[5-9][0-9][0-9][0-9]ms" logs/sentiment_analysis.log

# Check error patterns
grep "ERROR.*sentiment" logs/sentiment_analysis.log | tail -20
```

#### Monitoring Compliance
```bash
# Check GDPR request processing
grep "GDPR.*request.*completed" logs/sentiment_analysis.log

# Monitor data retention cleanup
grep "expired.*record.*deleted" logs/sentiment_analysis.log
```

## Performance Tuning

### Cache Optimization

```python
# Adaptive cache configuration
CACHE_PERFORMANCE_CONFIG = {
    "adaptive_sizing": True,
    "prefetch_popular_results": True,
    "compression_enabled": True,
    "ttl_based_on_confidence": True,
    "memory_threshold_mb": 512
}
```

### Database Optimization

```sql
-- PostgreSQL optimization for sentiment data
CREATE INDEX CONCURRENTLY idx_sentiment_timestamp ON sentiment_analysis_log (timestamp);
CREATE INDEX CONCURRENTLY idx_sentiment_user ON sentiment_analysis_log (user_id_hash);
CREATE INDEX CONCURRENTLY idx_sentiment_repository ON sentiment_analysis_log (repository);

-- Partitioning for large datasets
CREATE TABLE sentiment_analysis_log_2025 PARTITION OF sentiment_analysis_log 
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

### Auto-scaling Configuration

```python
AUTOSCALING_CONFIG = {
    "metrics_collection_interval": 30,  # seconds
    "scale_up_conditions": {
        "cpu_threshold": 80,
        "memory_threshold": 85,
        "queue_depth_threshold": 50,
        "response_time_threshold": 2000  # ms
    },
    "scale_down_conditions": {
        "cpu_threshold": 30,
        "memory_threshold": 40,
        "queue_depth_threshold": 5,
        "idle_time_minutes": 10
    }
}
```

## Backup and Recovery

### Data Backup Strategy

```bash
# Database backup
pg_dump healing_guard | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Redis backup
redis-cli --rdb backup_redis_$(date +%Y%m%d_%H%M%S).rdb

# Application data backup
tar -czf app_data_$(date +%Y%m%d_%H%M%S).tar.gz logs/ data/ config/
```

### Disaster Recovery Plan

1. **Database Recovery**: Restore from latest backup
2. **Cache Rebuild**: Allow cache to warm up naturally
3. **Service Restart**: Use health checks to verify recovery
4. **Data Validation**: Run compliance audits post-recovery

## Scaling Considerations

### Horizontal Scaling

- Deploy multiple instances behind a load balancer
- Use Redis for shared caching across instances
- Implement session affinity if stateful operations exist
- Consider database read replicas for high query loads

### Vertical Scaling

- Monitor resource usage patterns
- Scale CPU for compute-intensive sentiment analysis
- Scale memory for large caches and concurrent processing
- Scale storage for growing datasets and logs

## Compliance Maintenance

### Regular Tasks

1. **Data Retention Cleanup** (Daily)
```bash
# Automated cleanup via cron
0 2 * * * /app/scripts/cleanup_expired_data.sh
```

2. **Compliance Audits** (Weekly)
```bash
# Run comprehensive compliance check
0 0 * * 0 curl -X POST http://localhost:8000/admin/compliance/audit
```

3. **Privacy Policy Updates** (Quarterly)
- Review and update privacy policies
- Notify users of changes per compliance requirements
- Update consent mechanisms if needed

## Support and Maintenance

### Health Monitoring Commands

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/sentiment/stats

# Compliance status
curl http://localhost:8000/compliance/status

# Performance metrics
curl http://localhost:8000/metrics
```

### Maintenance Windows

Plan regular maintenance windows for:
- Security updates
- Compliance framework updates
- Performance optimizations
- Database maintenance

This deployment guide provides comprehensive instructions for successfully deploying the sentiment analysis system in production environments with proper monitoring, security, and compliance measures in place.