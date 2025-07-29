# Disaster Recovery Plan

## Executive Summary

This document outlines the disaster recovery (DR) procedures for the Self-Healing Pipeline Guard system. Our DR strategy ensures minimal downtime and data loss in the event of system failures, with Recovery Time Objective (RTO) of 15 minutes and Recovery Point Objective (RPO) of 5 minutes.

## Recovery Objectives

| Metric | Target | Business Impact |
|--------|--------|----------------|
| **RTO (Recovery Time Objective)** | 15 minutes | Acceptable downtime for critical services |
| **RPO (Recovery Point Objective)** | 5 minutes | Maximum acceptable data loss |
| **MTTR (Mean Time To Recovery)** | 10 minutes | Average time to restore services |
| **MTBF (Mean Time Between Failures)** | 720 hours | System reliability target |

## Disaster Scenarios

### Scenario Classifications

#### Category 1: Minor Incidents (< 5 minutes impact)
- Single container failure
- Temporary network connectivity issues  
- Individual service degradation
- **Response**: Automatic healing, monitoring alerts

#### Category 2: Major Incidents (5-30 minutes impact)
- Database connectivity loss
- Redis cache failure
- Application server crash
- Load balancer failure
- **Response**: Automated failover, manual intervention if needed

#### Category 3: Critical Disasters (30+ minutes impact)
- Complete data center outage
- Multi-zone cloud provider failure
- Catastrophic database corruption
- Security breach requiring system isolation
- **Response**: Full DR activation, executive escalation

## Infrastructure Overview

### Primary Environment
- **Cloud Provider**: Multi-cloud (AWS Primary, GCP Secondary)
- **Regions**: us-east-1 (primary), us-west-2 (DR)
- **Architecture**: Containerized microservices on Kubernetes
- **Data Stores**: PostgreSQL (primary), Redis (cache), S3 (object storage)

### Backup Infrastructure
- **Database Backups**: Automated hourly snapshots, cross-region replication
- **File Backups**: Real-time sync to S3 with cross-region replication
- **Configuration Backups**: GitOps-based infrastructure as code
- **Application Backups**: Container images in multi-region registries

## Recovery Procedures

### 1. Database Recovery

#### PostgreSQL Primary Failure

```bash
#!/bin/bash
# database-recovery.sh

# Step 1: Assess damage
pg_dump --host=primary-db --port=5432 --username=postgres --verbose --clean --no-owner --no-acl --format=custom healing_guard > /tmp/assessment.dump

# Step 2: Promote standby to primary
kubectl patch postgresql healing-guard-postgres -p '{"spec":{"postgresql":{"parameters":{"hot_standby":"off"}}}}'

# Step 3: Update application connection strings
kubectl set env deployment/healing-guard DATABASE_URL="postgresql://postgres:password@postgres-standby:5432/healing_guard"

# Step 4: Verify connectivity
kubectl exec -it deployment/healing-guard -- python -c "
import asyncpg
import asyncio
async def test_connection():
    conn = await asyncpg.connect('postgresql://postgres:password@postgres-standby:5432/healing_guard')
    result = await conn.fetchval('SELECT version()')
    print(f'Connected to: {result}')
    await conn.close()
asyncio.run(test_connection())
"

# Step 5: Monitor for data consistency
kubectl logs -f deployment/healing-guard | grep -i "database\|connection\|error"
```

#### Point-in-Time Recovery

```bash
#!/bin/bash
# point-in-time-recovery.sh

TARGET_TIME="2024-01-29 10:30:00"
BACKUP_ID="backup-20240129-103000"

# Step 1: Stop application
kubectl scale deployment/healing-guard --replicas=0

# Step 2: Restore from backup
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier healing-guard-prod \
    --target-db-instance-identifier healing-guard-recovery \
    --restore-time "$TARGET_TIME" \
    --db-subnet-group-name healing-guard-subnet-group \
    --publicly-accessible

# Step 3: Wait for restoration
while [[ $(aws rds describe-db-instances --db-instance-identifier healing-guard-recovery --query 'DBInstances[0].DBInstanceStatus' --output text) != "available" ]]; do
    echo "Waiting for database restoration..."
    sleep 30
done

# Step 4: Update connection and restart
kubectl set env deployment/healing-guard DATABASE_URL="postgresql://postgres:password@healing-guard-recovery.region.rds.amazonaws.com:5432/healing_guard"
kubectl scale deployment/healing-guard --replicas=3
```

### 2. Application Recovery

#### Complete Application Failure

```bash
#!/bin/bash
# application-recovery.sh

# Step 1: Check application health
kubectl get pods -l app=healing-guard
kubectl describe deployment/healing-guard

# Step 2: Rollback to last known good version
LAST_GOOD_VERSION=$(kubectl rollout history deployment/healing-guard | tail -2 | head -1 | awk '{print $1}')
kubectl rollout undo deployment/healing-guard --to-revision=$LAST_GOOD_VERSION

# Step 3: Scale up instances
kubectl scale deployment/healing-guard --replicas=5

# Step 4: Verify health checks
for i in {1..30}; do
    if kubectl get pods -l app=healing-guard | grep -q "Running"; then
        echo "Application pods are running"
        break
    fi
    echo "Waiting for pods to start... ($i/30)"
    sleep 10
done

# Step 5: Run smoke tests
kubectl exec -it deployment/healing-guard -- python -m pytest tests/smoke/ -v
```

### 3. Network and Load Balancer Recovery

#### Load Balancer Failure

```bash
#!/bin/bash
# loadbalancer-recovery.sh

# Step 1: Check load balancer status
aws elbv2 describe-load-balancers --names healing-guard-lb

# Step 2: Create emergency load balancer
aws elbv2 create-load-balancer \
    --name healing-guard-emergency-lb \
    --subnets subnet-12345678 subnet-87654321 \
    --security-groups sg-12345678 \
    --scheme internet-facing \
    --type application

# Step 3: Configure target group
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name healing-guard-emergency-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id vpc-12345678 \
    --health-check-path /health \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

# Step 4: Register targets
kubectl get pods -l app=healing-guard -o wide | awk 'NR>1 {print $6}' | while read ip; do
    aws elbv2 register-targets --target-group-arn $TARGET_GROUP_ARN --targets Id=$ip,Port=8000
done

# Step 5: Update DNS
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch '{
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "api.healing-guard.com",
            "Type": "CNAME",
            "TTL": 60,
            "ResourceRecords": [{"Value": "healing-guard-emergency-lb-123456789.us-east-1.elb.amazonaws.com"}]
        }
    }]
}'
```

### 4. Data Center Failover

#### Complete Regional Outage

```bash
#!/bin/bash
# regional-failover.sh

# Step 1: Activate DR region
export KUBECONFIG=/path/to/dr-cluster-config
export AWS_DEFAULT_REGION=us-west-2

# Step 2: Scale up DR infrastructure
kubectl apply -f k8s/dr-manifests/
kubectl scale deployment/healing-guard --replicas=5

# Step 3: Promote DR database to primary
aws rds promote-read-replica --db-instance-identifier healing-guard-dr

# Step 4: Update DNS to point to DR
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch '{
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "api.healing-guard.com",
            "Type": "CNAME",
            "TTL": 60,
            "ResourceRecords": [{"Value": "healing-guard-dr-lb.us-west-2.elb.amazonaws.com"}]
        }
    }]
}'

# Step 5: Notify stakeholders
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"🚨 DR Activated: Primary region down, failed over to us-west-2. RTO target: 15 minutes."}' \
    $SLACK_WEBHOOK_URL
```

## Monitoring and Alerting

### Health Check Endpoints

```python
# Application health checks
@app.get("/health")
async def health_check():
    checks = {
        "database": await check_database_connection(),
        "redis": await check_redis_connection(),
        "external_apis": await check_external_dependencies(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        },
        status_code=status_code
    )
```

### Critical Alerts

```yaml
# Disaster-level alerts
groups:
  - name: disaster.rules
    rules:
      # Complete service outage
      - alert: ServiceCompletelyDown
        expr: up{job="healing-guard"} == 0
        for: 2m
        labels:
          severity: disaster
        annotations:
          summary: "Complete service outage detected"
          runbook_url: "https://docs.company.com/runbooks/complete-outage"
          
      # Database cluster failure
      - alert: DatabaseClusterDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: disaster
        annotations:
          summary: "Database cluster completely unavailable"
          runbook_url: "https://docs.company.com/runbooks/database-disaster"
          
      # High error rate indicating system failure
      - alert: SystemFailureErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.5
        for: 3m
        labels:
          severity: disaster
        annotations:
          summary: "System failure - error rate exceeding 50%"
```

## Communication Plan

### Escalation Matrix

| Severity | Response Team | Notification Time | Stakeholders |
|----------|---------------|-------------------|--------------|
| **Category 1** | On-call Engineer | 5 minutes | Engineering team |
| **Category 2** | Senior Engineer + Manager | 10 minutes | Engineering + Product |
| **Category 3** | All hands + Executive | 15 minutes | Company-wide |

### Communication Channels

1. **Primary**: Slack #incident-response
2. **Secondary**: PagerDuty alerts
3. **Executive**: Direct phone calls
4. **Customer**: Status page updates
5. **External**: Twitter @HealingGuardStatus

### Status Page Updates

```bash
#!/bin/bash
# update-status-page.sh

INCIDENT_ID="inc-20240129-001"
STATUS="investigating"  # investigating, identified, monitoring, resolved
MESSAGE="We are investigating reports of service degradation."

curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
  -H "Authorization: OAuth TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incident": {
      "name": "Service Degradation",
      "status": "'$STATUS'",
      "impact_override": "major",
      "body": "'$MESSAGE'",
      "component_ids": ["COMPONENT_ID"],
      "incident_updates": [{
        "status": "'$STATUS'",
        "body": "'$MESSAGE'"
      }]
    }
  }'
```

## Testing and Validation

### Disaster Recovery Drills

#### Monthly Mini-Drills (15 minutes)
- Single component failure simulation
- Database failover testing
- Load balancer switching
- DNS failover validation

#### Quarterly Full Drills (2 hours)
- Complete regional failover
- Full data restoration
- End-to-end system validation
- Team coordination testing

#### Annual Chaos Engineering (4 hours)
- Multi-component failure simulation
- Network partition testing
- Security incident response
- Business continuity validation

### Validation Checklists

#### Post-Recovery Validation

```bash
#!/bin/bash
# post-recovery-validation.sh

echo "🔍 Starting post-recovery validation..."

# 1. Application health
echo "Checking application health..."
curl -f http://api.healing-guard.com/health || echo "❌ Health check failed"

# 2. Database connectivity
echo "Testing database connectivity..."
kubectl exec -it deployment/healing-guard -- python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('$DATABASE_URL')
    result = await conn.fetchval('SELECT COUNT(*) FROM pipelines')
    print(f'✅ Database accessible, {result} pipelines found')
    await conn.close()
asyncio.run(test())
"

# 3. Critical functionality
echo "Testing critical endpoints..."
curl -f -X POST http://api.healing-guard.com/api/v1/pipelines/test/analyze || echo "❌ Analysis endpoint failed"

# 4. Monitoring systems
echo "Checking monitoring..."
curl -f http://prometheus:9090/-/healthy || echo "❌ Prometheus unhealthy"
curl -f http://grafana:3000/api/health || echo "❌ Grafana unhealthy"

# 5. Performance validation
echo "Running performance validation..."
kubectl exec -it deployment/healing-guard -- python -m pytest tests/performance/ --duration=300

echo "✅ Post-recovery validation complete"
```

## Compliance and Auditing

### Audit Requirements

#### SOC 2 Compliance
- Document all DR procedures and tests
- Maintain access logs for emergency procedures
- Regular third-party DR assessment
- Annual penetration testing of DR systems

#### Regulatory Compliance
- **GDPR**: Data recovery procedures for EU customers
- **HIPAA**: Protected health information recovery protocols
- **PCI DSS**: Secure payment data recovery standards

### Documentation Requirements

1. **Incident Reports**: Detailed post-mortem for each activation
2. **Test Reports**: Results from all DR drills and validations  
3. **Change Control**: All DR procedure modifications documented
4. **Access Logs**: Emergency access and privilege escalations

## Continuous Improvement

### Post-Incident Review Process

1. **Immediate**: Hot wash within 24 hours
2. **Detailed**: Full post-mortem within 1 week
3. **Action Items**: Improvement tasks assigned and tracked
4. **Follow-up**: Implementation verification within 30 days

### Metrics and KPIs

| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| RTO Achievement | 95% | 92% | ↗️ |
| RPO Achievement | 98% | 96% | ↗️ |
| Drill Success Rate | 90% | 88% | ↗️ |
| False Positive Rate | <5% | 3% | ↘️ |

### Technology Roadmap

#### Q2 2024
- Implement automated DR orchestration
- Deploy multi-region active-active architecture
- Enhance real-time monitoring and alerting

#### Q3 2024
- Integrate AI-powered failure prediction
- Implement chaos engineering automation
- Deploy self-healing infrastructure

#### Q4 2024
- Achieve zero-downtime deployments
- Implement predictive disaster prevention
- Deploy quantum-resistant backup encryption

---

**Last Updated**: January 29, 2025  
**Next Review**: April 29, 2025  
**Document Owner**: Platform Engineering Team  
**Approved By**: CTO, Head of Engineering