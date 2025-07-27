# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents with Self-Healing Pipeline Guard. Follow these procedures to quickly identify, contain, and resolve issues.

## Incident Classification

### Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **P0 - Critical** | Complete service outage | 15 minutes | API completely down, database failure |
| **P1 - High** | Major functionality impaired | 1 hour | Healing strategies not working, webhook failures |
| **P2 - Medium** | Minor functionality impaired | 4 hours | Dashboard issues, delayed notifications |
| **P3 - Low** | Cosmetic or documentation issues | 24 hours | UI glitches, documentation errors |

### Incident Types

- **Service Outage**: Core services unavailable
- **Performance Degradation**: Slow response times or timeouts
- **Data Corruption**: Database or configuration corruption
- **Security Incident**: Unauthorized access or security breach
- **Integration Failure**: CI/CD platform integration issues
- **Resource Exhaustion**: Memory, CPU, or disk space issues

## Emergency Contacts

### On-Call Rotation

| Role | Primary | Secondary | Phone |
|------|---------|-----------|--------|
| **Incident Commander** | John Doe | Jane Smith | +1-555-0100 |
| **Engineering Lead** | Alice Johnson | Bob Wilson | +1-555-0101 |
| **Security Lead** | Charlie Brown | Diana Prince | +1-555-0102 |
| **Infrastructure Lead** | Eve Davis | Frank Miller | +1-555-0103 |

### Escalation Path

1. **L1 Support** → **L2 Engineering** → **Engineering Lead** → **VP Engineering**
2. **Security Issues** → **Security Lead** → **CISO** → **CEO**
3. **Legal/Compliance** → **Legal Team** → **Chief Legal Officer**

## P0 - Critical Incident Response

### Immediate Actions (0-15 minutes)

1. **Acknowledge the incident**
   ```bash
   # Update status page
   curl -X POST https://status.terragonlabs.com/api/incidents \
     -H "Authorization: Bearer $STATUS_PAGE_TOKEN" \
     -d '{"title": "Service Degradation", "status": "investigating"}'
   ```

2. **Assemble incident response team**
   - Page incident commander
   - Page engineering lead
   - Join incident channel: `#incident-response`

3. **Initial assessment**
   ```bash
   # Check service health
   curl https://healing-guard.terragonlabs.com/health
   
   # Check infrastructure status
   kubectl get pods -n healing-guard
   
   # Check recent deployments
   kubectl rollout history deployment/healing-guard -n healing-guard
   ```

4. **Establish communication**
   - Post in `#incident-response` channel
   - Start incident bridge call
   - Notify stakeholders

### Investigation Phase (15-30 minutes)

1. **Gather system information**
   ```bash
   # Check application logs
   kubectl logs -f deployment/healing-guard -n healing-guard --tail=100
   
   # Check system metrics
   curl https://prometheus.terragonlabs.com/api/v1/query?query=up{job="healing-guard"}
   
   # Check database connectivity
   kubectl exec -it deployment/postgres -n healing-guard -- psql -U healing_user -c "SELECT 1;"
   ```

2. **Check monitoring dashboards**
   - Grafana: `https://grafana.terragonlabs.com/d/healing-guard-overview`
   - Prometheus: `https://prometheus.terragonlabs.com`
   - Application metrics: Error rates, response times, throughput

3. **Review recent changes**
   ```bash
   # Check recent deployments
   kubectl describe deployment healing-guard -n healing-guard
   
   # Check recent code changes
   git log --oneline --since="2 hours ago"
   
   # Check configuration changes
   kubectl get configmap healing-guard-config -o yaml
   ```

### Containment Actions

1. **If recent deployment caused the issue**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/healing-guard -n healing-guard
   
   # Verify rollback
   kubectl rollout status deployment/healing-guard -n healing-guard
   ```

2. **If database issues**
   ```bash
   # Check database connections
   kubectl get pods -l app=postgres -n healing-guard
   
   # Restart database if needed
   kubectl delete pod -l app=postgres -n healing-guard
   
   # Check database logs
   kubectl logs -f deployment/postgres -n healing-guard
   ```

3. **If resource exhaustion**
   ```bash
   # Scale up application
   kubectl scale deployment healing-guard --replicas=6 -n healing-guard
   
   # Add more resources
   kubectl patch deployment healing-guard -n healing-guard -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"2Gi","cpu":"1000m"}}}]}}}}'
   ```

### Resolution Phase

1. **Verify fix**
   ```bash
   # Test critical endpoints
   curl -f https://healing-guard.terragonlabs.com/health
   curl -f https://healing-guard.terragonlabs.com/api/v1/status
   
   # Run smoke tests
   kubectl apply -f tests/smoke-test.yaml
   ```

2. **Monitor recovery**
   - Watch error rates return to normal
   - Verify healing functionality is restored
   - Check customer impact metrics

3. **Update status page**
   ```bash
   curl -X PATCH https://status.terragonlabs.com/api/incidents/$INCIDENT_ID \
     -H "Authorization: Bearer $STATUS_PAGE_TOKEN" \
     -d '{"status": "resolved", "message": "Issue has been resolved"}'
   ```

## P1 - High Severity Response

### Common P1 Scenarios

#### Healing Strategies Not Working

1. **Check strategy engine**
   ```bash
   # Check strategy engine logs
   kubectl logs -f deployment/healing-guard -n healing-guard | grep strategy
   
   # Check strategy configuration
   curl https://healing-guard.terragonlabs.com/api/v1/strategies/status
   ```

2. **Verify ML model health**
   ```bash
   # Check model loading
   curl https://healing-guard.terragonlabs.com/api/v1/ml/health
   
   # Test model inference
   curl -X POST https://healing-guard.terragonlabs.com/api/v1/ml/predict \
     -H "Content-Type: application/json" \
     -d '{"logs": ["test error message"]}'
   ```

3. **Restart strategy engine if needed**
   ```bash
   kubectl delete pod -l component=strategy-engine -n healing-guard
   ```

#### Webhook Processing Failures

1. **Check webhook endpoints**
   ```bash
   # Test webhook endpoints
   curl -X POST https://healing-guard.terragonlabs.com/webhooks/github \
     -H "Content-Type: application/json" \
     -H "X-GitHub-Event: workflow_run" \
     -d '{"action": "completed", "workflow_run": {"conclusion": "failure"}}'
   ```

2. **Check webhook queue**
   ```bash
   # Check Redis queue depth
   kubectl exec -it deployment/redis -n healing-guard -- redis-cli llen webhook_queue
   
   # Check worker status
   kubectl logs -f deployment/webhook-worker -n healing-guard
   ```

3. **Scale webhook workers if needed**
   ```bash
   kubectl scale deployment webhook-worker --replicas=5 -n healing-guard
   ```

## Performance Issues

### High Response Times

1. **Check application performance**
   ```bash
   # Check response time metrics
   curl 'https://prometheus.terragonlabs.com/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket{job="healing-guard"}[5m]))'
   
   # Check database query performance
   kubectl exec -it deployment/postgres -n healing-guard -- psql -U healing_user -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
   ```

2. **Check resource utilization**
   ```bash
   # Check CPU and memory usage
   kubectl top pods -n healing-guard
   
   # Check database connections
   kubectl exec -it deployment/postgres -n healing-guard -- psql -U healing_user -c "SELECT count(*) FROM pg_stat_activity;"
   ```

3. **Scale resources if needed**
   ```bash
   # Scale application horizontally
   kubectl scale deployment healing-guard --replicas=8 -n healing-guard
   
   # Scale database resources
   kubectl patch postgresql healing-guard-db -p '{"spec":{"resources":{"requests":{"memory":"4Gi","cpu":"2000m"}}}}'
   ```

### Memory Leaks

1. **Identify memory usage patterns**
   ```bash
   # Check memory usage over time
   curl 'https://prometheus.terragonlabs.com/api/v1/query_range?query=process_resident_memory_bytes{job="healing-guard"}&start=2024-01-01T00:00:00Z&end=2024-01-01T12:00:00Z&step=300s'
   
   # Check for memory leaks in application
   kubectl exec -it deployment/healing-guard -n healing-guard -- python -c "
   import psutil
   import gc
   print(f'Memory usage: {psutil.virtual_memory().percent}%')
   print(f'Objects in memory: {len(gc.get_objects())}')
   "
   ```

2. **Force garbage collection**
   ```bash
   # Trigger garbage collection
   curl -X POST https://healing-guard.terragonlabs.com/api/v1/admin/gc
   ```

3. **Restart pods if memory usage is critical**
   ```bash
   kubectl delete pod -l app=healing-guard -n healing-guard
   ```

## Data Issues

### Database Corruption

1. **Check database integrity**
   ```bash
   kubectl exec -it deployment/postgres -n healing-guard -- psql -U healing_user -c "
   SELECT schemaname, tablename, attname, n_distinct, correlation 
   FROM pg_stats 
   WHERE schemaname = 'public';
   "
   ```

2. **Restore from backup if needed**
   ```bash
   # Stop application
   kubectl scale deployment healing-guard --replicas=0 -n healing-guard
   
   # Restore database from latest backup
   kubectl exec -it deployment/postgres -n healing-guard -- pg_restore -U healing_user -d healing_guard /backups/latest.dump
   
   # Restart application
   kubectl scale deployment healing-guard --replicas=3 -n healing-guard
   ```

### Configuration Corruption

1. **Backup current configuration**
   ```bash
   kubectl get configmap healing-guard-config -o yaml > config-backup-$(date +%s).yaml
   ```

2. **Restore known good configuration**
   ```bash
   kubectl apply -f config/known-good-config.yaml
   kubectl rollout restart deployment/healing-guard -n healing-guard
   ```

## Security Incidents

### Unauthorized Access

1. **Immediate containment**
   ```bash
   # Check active sessions
   curl https://healing-guard.terragonlabs.com/api/v1/admin/sessions
   
   # Revoke all API tokens
   curl -X POST https://healing-guard.terragonlabs.com/api/v1/admin/revoke-all-tokens
   
   # Change all passwords
   kubectl create secret generic healing-guard-secrets \
     --from-literal=jwt-secret=$(openssl rand -hex 32) \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

2. **Investigate breach**
   ```bash
   # Check access logs
   kubectl logs deployment/healing-guard -n healing-guard | grep -E "(401|403|login|auth)"
   
   # Check unusual API usage
   curl 'https://prometheus.terragonlabs.com/api/v1/query?query=rate(http_requests_total{status="200"}[5m]) > 100'
   ```

3. **Notify security team**
   - Send alert to `#security-incidents`
   - Page security lead
   - Document timeline of events

## Communication Templates

### Initial Incident Notification

```
🚨 INCIDENT ALERT - P0
Title: [Brief description]
Impact: [Customer/service impact]
Status: Investigating
ETA: [Estimated resolution time]
Updates: Will update every 15 minutes
Incident Commander: [Name]
```

### Status Updates

```
📊 INCIDENT UPDATE - P0
Status: [Investigating/Identified/Resolving/Resolved]
Progress: [What's been done]
Next Steps: [What's being done next]
ETA: [Updated ETA]
Time to next update: 15 minutes
```

### Resolution Notification

```
✅ INCIDENT RESOLVED - P0
Resolution: [What fixed the issue]
Root Cause: [Brief root cause]
Prevention: [What will prevent recurrence]
Duration: [Total incident duration]
Post-Mortem: [Link to post-mortem document]
```

## Post-Incident Actions

### Immediate (0-24 hours)

1. **Document the incident**
   - Create incident report
   - Collect all logs and metrics
   - Document timeline of events
   - Note all actions taken

2. **Verify full restoration**
   - Run comprehensive tests
   - Check all integrations
   - Verify monitoring is working
   - Confirm customer impact resolved

### Short-term (1-7 days)

1. **Conduct post-mortem**
   - Schedule blameless post-mortem
   - Identify root cause
   - Create action items
   - Share learnings with team

2. **Implement immediate fixes**
   - Address identified gaps
   - Improve monitoring
   - Update documentation
   - Strengthen preventive measures

### Long-term (1-4 weeks)

1. **Implement systemic improvements**
   - Architecture changes
   - Process improvements
   - Training updates
   - Tool enhancements

2. **Update procedures**
   - Revise runbooks
   - Update escalation procedures
   - Improve automation
   - Enhance testing

---

**Remember**: Stay calm, communicate clearly, and focus on resolution. Every incident is a learning opportunity to improve our systems and processes.