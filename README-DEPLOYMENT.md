# Self-Healing Pipeline Guard - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Self-Healing Pipeline Guard in production environments using Kubernetes, Docker Compose, or standalone deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Docker Compose Deployment](#docker-compose-deployment)
5. [Helm Chart Deployment](#helm-chart-deployment)
6. [Configuration](#configuration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Memory**: Minimum 4GB RAM, Recommended 8GB+ RAM
- **Storage**: Minimum 20GB, Recommended 100GB+ SSD
- **Network**: HTTPS/TLS termination capability

### Software Dependencies

- **Docker**: Version 20.10+ (for containerized deployment)
- **Kubernetes**: Version 1.24+ (for K8s deployment)
- **Helm**: Version 3.10+ (for Helm deployment)
- **PostgreSQL**: Version 12+ (database)
- **Redis**: Version 6+ (caching and queuing)

### External Services

- **CI/CD Platform**: GitHub Actions, GitLab CI, Jenkins, etc.
- **Container Registry**: Docker Hub, AWS ECR, GCR, etc.
- **DNS Provider**: For domain configuration
- **SSL/TLS Certificates**: Let's Encrypt or commercial certificates

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/terragon-labs/self-healing-pipeline-guard.git
cd self-healing-pipeline-guard

# Copy and customize environment configuration
cp .env.example .env
vim .env  # Configure your environment variables
```

### 2. Deploy with Docker Compose (Development/Testing)

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f healing-guard
```

### 3. Deploy with Kubernetes (Production)

```bash
# Deploy using the deployment script
chmod +x deploy/deploy.sh
./deploy/deploy.sh deploy

# Or manually with kubectl
kubectl apply -f k8s/
```

## Kubernetes Deployment

### Architecture Overview

The Kubernetes deployment includes:

- **Main Application**: 3+ replicas with auto-scaling
- **PostgreSQL**: Persistent database with backup
- **Redis**: In-memory cache and message queue
- **NGINX Ingress**: Load balancing and SSL termination
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting

### Step-by-Step Deployment

#### 1. Prepare Kubernetes Cluster

```bash
# Ensure kubectl is configured
kubectl cluster-info

# Create namespace
kubectl create namespace healing-guard

# Set context
kubectl config set-context --current --namespace=healing-guard
```

#### 2. Configure Secrets

```bash
# Create TLS secret for HTTPS
kubectl create secret tls healing-guard-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n healing-guard

# Create application secrets
kubectl create secret generic healing-guard-secrets \
  --from-literal=DATABASE_PASSWORD=your_db_password \
  --from-literal=REDIS_PASSWORD=your_redis_password \
  --from-literal=JWT_SECRET_KEY=your_jwt_secret \
  -n healing-guard
```

#### 3. Deploy Application

```bash
# Deploy all components
kubectl apply -f k8s/

# Wait for deployment
kubectl wait --for=condition=available --timeout=600s deployment/healing-guard -n healing-guard

# Check status
kubectl get all -n healing-guard
```

#### 4. Configure Ingress

```bash
# Update ingress with your domain
sed -i 's/healing-guard.terragonlabs.com/your-domain.com/g' k8s/ingress.yaml
kubectl apply -f k8s/ingress.yaml
```

### Auto-Scaling Configuration

The deployment includes Horizontal Pod Autoscaler (HPA):

```yaml
# Current HPA settings
minReplicas: 2
maxReplicas: 20
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Storage Configuration

Persistent volumes are configured for:

- **Application Data**: 10GB (configurable)
- **PostgreSQL**: 20GB (configurable)
- **Redis**: 5GB (configurable)
- **Prometheus**: 10GB (configurable)

## Docker Compose Deployment

### Production Docker Compose

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale the application
docker-compose up -d --scale healing-guard=3

# Enable monitoring stack
docker-compose --profile monitoring up -d
```

### Environment Configuration

Create `.env` file:

```bash
# Application
HEALING_GUARD_ENV=production
HEALING_GUARD_LOG_LEVEL=INFO

# Database
POSTGRES_PASSWORD=secure_password
DATABASE_URL=postgresql://healing_guard:secure_password@postgres:5432/healing_guard

# Redis
REDIS_PASSWORD=redis_secure_password
REDIS_URL=redis://:redis_secure_password@redis:6379/0

# Security
JWT_SECRET_KEY=your_very_secure_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# External Services
GITHUB_WEBHOOK_SECRET=your_github_webhook_secret
SLACK_BOT_TOKEN=your_slack_bot_token
```

## Helm Chart Deployment

### Install with Helm

```bash
# Add required repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install the chart
helm install healing-guard ./helm/healing-guard \
  --namespace healing-guard \
  --create-namespace \
  --set image.tag=latest \
  --set ingress.hosts[0].host=your-domain.com \
  --set postgresql.auth.password=secure_password \
  --set redis.auth.password=redis_password
```

### Customize Values

Create `values-production.yaml`:

```yaml
# Production values
replicaCount: 3

image:
  tag: "v1.0.0"

ingress:
  enabled: true
  hosts:
    - host: healing-guard.your-domain.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 50

persistence:
  enabled: true
  size: 100Gi
  storageClass: fast-ssd
```

Deploy with custom values:

```bash
helm install healing-guard ./helm/healing-guard \
  -f values-production.yaml \
  --namespace healing-guard
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HEALING_GUARD_ENV` | Environment (dev/staging/production) | `production` | Yes |
| `HEALING_GUARD_LOG_LEVEL` | Log level (DEBUG/INFO/WARNING/ERROR) | `INFO` | No |
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `JWT_SECRET_KEY` | JWT signing key | - | Yes |
| `GITHUB_WEBHOOK_SECRET` | GitHub webhook secret | - | No |
| `SLACK_BOT_TOKEN` | Slack integration token | - | No |

### Database Configuration

```sql
-- Create database and user
CREATE DATABASE healing_guard;
CREATE USER healing_guard WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE healing_guard TO healing_guard;

-- Enable required extensions
\c healing_guard
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

### Redis Configuration

```bash
# Redis configuration
requirepass redis_secure_password
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Monitoring & Observability

### Prometheus Metrics

Available metrics endpoints:

- **Application**: `http://localhost:9090/metrics`
- **System**: Node Exporter metrics
- **Database**: PostgreSQL Exporter metrics
- **Cache**: Redis Exporter metrics

### Grafana Dashboards

Pre-configured dashboards:

1. **Application Overview**: Request rates, response times, error rates
2. **System Resources**: CPU, memory, disk, network
3. **Database Performance**: Query performance, connection pools
4. **Auto-Scaling**: HPA status, scaling events
5. **Security**: Failed authentication attempts, security events

### Health Checks

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Basic health check | 200 OK |
| `/health/ready` | Readiness probe | 200 OK |
| `/health/live` | Liveness probe | 200 OK |
| `/metrics` | Prometheus metrics | 200 OK |

### Logging

Logs are structured in JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "healing_guard.core.quantum_planner",
  "message": "Task optimization completed",
  "correlation_id": "req_123456",
  "user_id": "user_789",
  "execution_time": 0.245
}
```

## Security

### Network Security

- **TLS 1.3**: All external communication encrypted
- **Network Policies**: Kubernetes network isolation
- **Firewall Rules**: Restrict unnecessary ports
- **VPN Access**: Administrative access via VPN

### Application Security

- **Authentication**: JWT-based with refresh tokens
- **Authorization**: Role-based access control (RBAC)
- **Input Validation**: All inputs sanitized and validated
- **Rate Limiting**: Per-user and per-IP rate limits
- **Security Headers**: OWASP-recommended headers

### Data Security

- **Encryption at Rest**: Database and file encryption
- **Encryption in Transit**: TLS for all communications
- **Secrets Management**: Kubernetes secrets or external vault
- **GDPR Compliance**: Built-in data governance

### Container Security

- **Non-root User**: Containers run as non-root
- **Read-only Filesystem**: Where possible
- **Security Scanning**: Trivy, Clair, or similar
- **Image Signing**: Cosign or Notary v2

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

```bash
# Check logs
kubectl logs -f deployment/healing-guard -n healing-guard

# Check events
kubectl get events -n healing-guard --sort-by='.lastTimestamp'

# Check configuration
kubectl describe configmap healing-guard-config -n healing-guard
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/healing-guard -n healing-guard -- \
  python -c "import asyncpg; print('DB connection test')"

# Check database logs
kubectl logs -f statefulset/postgres -n healing-guard
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top pods -n healing-guard

# Check HPA status
kubectl get hpa -n healing-guard

# Review metrics
curl http://healing-guard.your-domain.com/metrics
```

#### 4. SSL/TLS Issues

```bash
# Check certificate
openssl s_client -connect healing-guard.your-domain.com:443

# Check ingress
kubectl describe ingress healing-guard-ingress -n healing-guard

# Check cert-manager (if used)
kubectl get certificates -n healing-guard
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Temporary debug mode
kubectl set env deployment/healing-guard HEALING_GUARD_LOG_LEVEL=DEBUG -n healing-guard

# Revert to normal logging
kubectl set env deployment/healing-guard HEALING_GUARD_LOG_LEVEL=INFO -n healing-guard
```

## Maintenance

### Backup Procedures

#### Database Backup

```bash
# Automated backup script
kubectl create job --from=cronjob/postgres-backup backup-$(date +%Y%m%d) -n healing-guard

# Manual backup
kubectl exec -it postgres-0 -n healing-guard -- \
  pg_dump -U healing_guard healing_guard > backup-$(date +%Y%m%d).sql
```

#### Application Data Backup

```bash
# Backup persistent volumes
kubectl exec -it deployment/healing-guard -n healing-guard -- \
  tar -czf /tmp/app-backup-$(date +%Y%m%d).tar.gz /app/data
```

### Updates and Rollouts

#### Rolling Update

```bash
# Update image version
kubectl set image deployment/healing-guard healing-guard=terragonlabs/healing-guard:v1.1.0 -n healing-guard

# Check rollout status
kubectl rollout status deployment/healing-guard -n healing-guard

# Rollback if needed
kubectl rollout undo deployment/healing-guard -n healing-guard
```

#### Using Helm

```bash
# Update with Helm
helm upgrade healing-guard ./helm/healing-guard \
  --set image.tag=v1.1.0 \
  --namespace healing-guard

# Rollback with Helm
helm rollback healing-guard -n healing-guard
```

### Scaling Operations

#### Manual Scaling

```bash
# Scale pods
kubectl scale deployment healing-guard --replicas=5 -n healing-guard

# Scale database resources
kubectl patch statefulset postgres -p '{"spec":{"template":{"spec":{"containers":[{"name":"postgres","resources":{"limits":{"cpu":"2","memory":"4Gi"}}}]}}}}' -n healing-guard
```

#### Auto-Scaling Adjustment

```bash
# Update HPA thresholds
kubectl patch hpa healing-guard-hpa -p '{"spec":{"targetCPUUtilizationPercentage":60}}' -n healing-guard
```

### Monitoring and Alerts

#### Key Metrics to Monitor

1. **Application Health**: Response time, error rate, availability
2. **Resource Usage**: CPU, memory, disk, network
3. **Database Performance**: Connection pool, query time
4. **Auto-Scaling Events**: Scale up/down events
5. **Security Events**: Authentication failures, suspicious activity

#### Alert Thresholds

- **High Error Rate**: >5% for 5 minutes
- **High Response Time**: >2s for 5 minutes
- **High CPU Usage**: >80% for 10 minutes
- **High Memory Usage**: >85% for 10 minutes
- **Database Connections**: >80% of max for 5 minutes

### Support and Documentation

- **GitHub Issues**: https://github.com/terragon-labs/self-healing-pipeline-guard/issues
- **Documentation**: https://docs.terragonlabs.com/healing-guard
- **Support Email**: support@terragonlabs.com
- **Community Discord**: https://discord.gg/terragon-labs

---

## License

This deployment guide is part of the Self-Healing Pipeline Guard project, licensed under the MIT License.