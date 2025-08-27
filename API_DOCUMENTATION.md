# Healing Guard API Documentation

## Overview

The Healing Guard API provides comprehensive endpoints for managing self-healing pipeline operations, monitoring, security, compliance, and multi-tenant management.

## Base URL
```
Production: https://api.healingguard.terragon.ai
Development: http://localhost:8000
```

## Authentication

All API endpoints require JWT authentication unless otherwise specified.

```http
Authorization: Bearer <jwt_token>
```

## Core API Endpoints

### Health and Status

#### GET /api/health
Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-27T21:01:56Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "monitoring": "healthy"
  }
}
```

#### GET /api/status
Returns detailed system status and metrics.

**Response:**
```json
{
  "system_status": "operational",
  "uptime": 86400,
  "active_healers": 5,
  "pending_alerts": 2,
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1
  }
}
```

### Pipeline Management

#### GET /api/pipelines
List all monitored pipelines.

**Query Parameters:**
- `tenant_id` (optional): Filter by tenant
- `status` (optional): Filter by status
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "pipelines": [
    {
      "id": "pipeline-123",
      "name": "Production Deployment",
      "status": "healthy",
      "last_run": "2025-08-27T20:45:00Z",
      "failure_count": 0,
      "tenant_id": "tenant-abc"
    }
  ],
  "total": 25,
  "limit": 50,
  "offset": 0
}
```

#### POST /api/pipelines
Register a new pipeline for monitoring.

**Request Body:**
```json
{
  "name": "New Pipeline",
  "webhook_url": "https://ci.example.com/webhook",
  "config": {
    "retry_attempts": 3,
    "timeout": 300,
    "notification_channels": ["slack", "email"]
  }
}
```

#### GET /api/pipelines/{pipeline_id}
Get detailed pipeline information.

**Response:**
```json
{
  "id": "pipeline-123",
  "name": "Production Deployment",
  "status": "healthy",
  "config": {...},
  "metrics": {
    "success_rate": 98.5,
    "avg_duration": 145.2,
    "total_runs": 1250
  },
  "recent_failures": []
}
```

### Failure Detection and Healing

#### GET /api/failures
List detected failures.

**Query Parameters:**
- `pipeline_id` (optional): Filter by pipeline
- `severity` (optional): Filter by severity level
- `status` (optional): Filter by resolution status
- `from_date`, `to_date` (optional): Date range filter

**Response:**
```json
{
  "failures": [
    {
      "id": "failure-456",
      "pipeline_id": "pipeline-123",
      "type": "build_failure",
      "severity": "high",
      "detected_at": "2025-08-27T20:30:00Z",
      "resolved_at": "2025-08-27T20:32:15Z",
      "healing_actions": [
        {
          "action": "retry_build",
          "result": "success",
          "duration": 135
        }
      ]
    }
  ]
}
```

#### POST /api/failures/{failure_id}/heal
Trigger manual healing for a specific failure.

**Request Body:**
```json
{
  "action": "custom_heal",
  "parameters": {
    "force_rebuild": true,
    "notify_team": true
  }
}
```

### Real-time Dashboard

#### WebSocket: /ws/dashboard
Real-time dashboard data stream.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.healingguard.terragon.ai/ws/dashboard');
```

**Message Types:**
- `system_metrics`: Real-time system performance data
- `pipeline_updates`: Pipeline status changes
- `failure_alerts`: New failure notifications
- `healing_progress`: Healing action progress

### ML and Analytics

#### GET /api/analytics/patterns
Get failure pattern analysis.

**Response:**
```json
{
  "patterns": [
    {
      "pattern_id": "pattern-789",
      "type": "build_timeout",
      "frequency": 15,
      "last_occurrence": "2025-08-27T19:45:00Z",
      "predicted_next": "2025-08-28T08:30:00Z",
      "confidence": 0.87
    }
  ]
}
```

#### POST /api/ml/predict
Request failure prediction for a specific pipeline.

**Request Body:**
```json
{
  "pipeline_id": "pipeline-123",
  "prediction_window": 24
}
```

**Response:**
```json
{
  "pipeline_id": "pipeline-123",
  "prediction": {
    "failure_probability": 0.23,
    "most_likely_failure": "dependency_timeout",
    "confidence": 0.91,
    "recommended_actions": [
      "increase_timeout_limit",
      "add_dependency_health_check"
    ]
  }
}
```

## Security and Compliance

### Authentication

#### POST /auth/login
Authenticate user and obtain JWT token.

**Request Body:**
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 3600,
  "user_info": {
    "id": "user-123",
    "role": "admin",
    "tenant_id": "tenant-abc"
  }
}
```

#### POST /auth/refresh
Refresh JWT token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Compliance Auditing

#### GET /api/compliance/audits
List compliance audit records.

**Query Parameters:**
- `standard` (optional): Compliance standard (sox, gdpr, hipaa, pci_dss)
- `status` (optional): Audit status
- `from_date`, `to_date` (optional): Date range

**Response:**
```json
{
  "audits": [
    {
      "id": "audit-789",
      "standard": "sox",
      "status": "compliant",
      "audit_date": "2025-08-27T00:00:00Z",
      "findings": [],
      "risk_score": 2.3,
      "digital_signature": "sha256:abc123..."
    }
  ]
}
```

#### POST /api/compliance/audits
Create new compliance audit.

**Request Body:**
```json
{
  "standard": "gdpr",
  "scope": ["data_processing", "user_consent"],
  "auditor_id": "auditor-456"
}
```

## Multi-Tenant Management

### Tenant Operations

#### GET /api/tenants
List all tenants (admin only).

**Response:**
```json
{
  "tenants": [
    {
      "id": "tenant-abc",
      "name": "Acme Corporation",
      "status": "active",
      "created_at": "2025-01-15T10:00:00Z",
      "resource_quota": {
        "max_pipelines": 100,
        "max_storage_gb": 500,
        "max_users": 50
      },
      "current_usage": {
        "pipelines": 23,
        "storage_gb": 125.5,
        "users": 12
      }
    }
  ]
}
```

#### POST /api/tenants
Create new tenant.

**Request Body:**
```json
{
  "name": "New Company",
  "admin_email": "admin@newcompany.com",
  "resource_quota": {
    "max_pipelines": 50,
    "max_storage_gb": 250,
    "max_users": 25
  }
}
```

### Resource Management

#### GET /api/tenants/{tenant_id}/usage
Get tenant resource usage statistics.

**Response:**
```json
{
  "tenant_id": "tenant-abc",
  "usage_period": {
    "start": "2025-08-01T00:00:00Z",
    "end": "2025-08-31T23:59:59Z"
  },
  "metrics": {
    "pipeline_executions": 1250,
    "healing_actions": 45,
    "data_processed_gb": 78.5,
    "api_calls": 15420
  },
  "billing_amount": 299.99
}
```

## Distributed Operations

### Cluster Management

#### GET /api/cluster/status
Get cluster status and node information.

**Response:**
```json
{
  "cluster_id": "cluster-main",
  "leader_node": "node-01",
  "total_nodes": 5,
  "healthy_nodes": 5,
  "nodes": [
    {
      "id": "node-01",
      "status": "healthy",
      "role": "leader",
      "load": 45.2,
      "last_heartbeat": "2025-08-27T21:01:55Z"
    }
  ]
}
```

#### POST /api/cluster/rebalance
Trigger cluster rebalancing.

**Request Body:**
```json
{
  "strategy": "even_distribution",
  "force": false
}
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "PIPELINE_NOT_FOUND",
    "message": "Pipeline with ID 'pipeline-123' not found",
    "details": {
      "resource_type": "pipeline",
      "resource_id": "pipeline-123"
    },
    "timestamp": "2025-08-27T21:01:56Z"
  }
}
```

### HTTP Status Codes

- `200` - OK
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## Rate Limiting

API endpoints are rate limited per tenant:

- **Standard endpoints:** 1000 requests per minute
- **Analytics endpoints:** 100 requests per minute
- **WebSocket connections:** 10 concurrent per user

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1693164116
```

## SDK and Libraries

### Python SDK
```bash
pip install healing-guard-sdk
```

```python
from healing_guard import HealingGuardClient

client = HealingGuardClient(
    base_url="https://api.healingguard.terragon.ai",
    api_token="your_jwt_token"
)

# List pipelines
pipelines = client.pipelines.list()

# Get real-time updates
client.dashboard.subscribe(callback=handle_update)
```

### JavaScript SDK
```bash
npm install @terragon/healing-guard-sdk
```

```javascript
import { HealingGuardClient } from '@terragon/healing-guard-sdk';

const client = new HealingGuardClient({
  baseUrl: 'https://api.healingguard.terragon.ai',
  apiToken: 'your_jwt_token'
});

// List pipelines
const pipelines = await client.pipelines.list();

// WebSocket connection
client.dashboard.connect({
  onSystemMetrics: (data) => console.log(data),
  onFailureAlert: (alert) => console.log(alert)
});
```

## Support

For API support and documentation updates:
- **Documentation:** https://docs.terragon.ai/healing-guard/api
- **Support:** support@terragon.ai
- **Status Page:** https://status.terragon.ai

---
*API Documentation v4.0 - Generated for Healing Guard Enterprise*