"""Multi-tenant isolation and resource management system.

Provides secure tenant isolation, resource quotas, and billing management
for enterprise multi-tenant deployments.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from ..security.enterprise_security import SecurityContext, security_manager
from ..monitoring.enhanced_monitoring import enhanced_monitoring
from ..compliance.advanced_audit import compliance_auditor, ComplianceStandard

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Tenant subscription tiers."""
    BASIC = "basic"
    STANDARD = "standard" 
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ResourceType(Enum):
    """Types of resources that can be limited."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    STORAGE_GB = "storage_gb"
    HEALING_OPERATIONS = "healing_operations"
    API_REQUESTS = "api_requests"
    CONCURRENT_JOBS = "concurrent_jobs"
    USERS = "users"
    AUDIT_RETENTION_DAYS = "audit_retention_days"


class TenantStatus(Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"
    DELINQUENT = "delinquent"


@dataclass
class ResourceQuota:
    """Resource quota definition."""
    resource_type: ResourceType
    limit: float
    used: float = 0.0
    unit: str = ""
    
    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        return (self.used / self.limit * 100) if self.limit > 0 else 0.0
    
    @property
    def available(self) -> float:
        """Get available resource amount."""
        return max(0.0, self.limit - self.used)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource_type": self.resource_type.value,
            "limit": self.limit,
            "used": self.used,
            "available": self.available,
            "utilization": self.utilization,
            "unit": self.unit
        }


@dataclass
class TenantConfig:
    """Tenant configuration and metadata."""
    tenant_id: str
    name: str
    tier: TenantTier
    status: TenantStatus
    created_at: datetime
    admin_email: str
    billing_contact: str
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "tier": self.tier.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "admin_email": self.admin_email,
            "billing_contact": self.billing_contact,
            "compliance_standards": [std.value for std in self.compliance_standards],
            "custom_config": self.custom_config,
            "tags": self.tags,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class ResourceUsageRecord:
    """Resource usage tracking record."""
    tenant_id: str
    resource_type: ResourceType
    amount: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class BillingRecord:
    """Billing and usage record."""
    id: str
    tenant_id: str
    billing_period_start: datetime
    billing_period_end: datetime
    resource_usage: Dict[ResourceType, float]
    total_cost: float
    currency: str = "USD"
    status: str = "pending"
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "billing_period_start": self.billing_period_start.isoformat(),
            "billing_period_end": self.billing_period_end.isoformat(),
            "resource_usage": {
                resource_type.value: usage 
                for resource_type, usage in self.resource_usage.items()
            },
            "total_cost": self.total_cost,
            "currency": self.currency,
            "status": self.status,
            "generated_at": self.generated_at.isoformat()
        }


class ResourceManager:
    """Manages resource quotas and usage tracking."""
    
    def __init__(self):
        self.quotas: Dict[str, Dict[ResourceType, ResourceQuota]] = {}
        self.usage_records: List[ResourceUsageRecord] = []
        self.tier_limits = self._initialize_tier_limits()
        self.lock = threading.RLock()
    
    def _initialize_tier_limits(self) -> Dict[TenantTier, Dict[ResourceType, float]]:
        """Initialize default resource limits by tier."""
        return {
            TenantTier.BASIC: {
                ResourceType.CPU_CORES: 2.0,
                ResourceType.MEMORY_GB: 4.0,
                ResourceType.STORAGE_GB: 10.0,
                ResourceType.HEALING_OPERATIONS: 100.0,
                ResourceType.API_REQUESTS: 1000.0,
                ResourceType.CONCURRENT_JOBS: 2.0,
                ResourceType.USERS: 5.0,
                ResourceType.AUDIT_RETENTION_DAYS: 30.0
            },
            TenantTier.STANDARD: {
                ResourceType.CPU_CORES: 4.0,
                ResourceType.MEMORY_GB: 8.0,
                ResourceType.STORAGE_GB: 50.0,
                ResourceType.HEALING_OPERATIONS: 500.0,
                ResourceType.API_REQUESTS: 5000.0,
                ResourceType.CONCURRENT_JOBS: 5.0,
                ResourceType.USERS: 25.0,
                ResourceType.AUDIT_RETENTION_DAYS: 90.0
            },
            TenantTier.PREMIUM: {
                ResourceType.CPU_CORES: 8.0,
                ResourceType.MEMORY_GB: 16.0,
                ResourceType.STORAGE_GB: 200.0,
                ResourceType.HEALING_OPERATIONS: 2000.0,
                ResourceType.API_REQUESTS: 25000.0,
                ResourceType.CONCURRENT_JOBS: 10.0,
                ResourceType.USERS: 100.0,
                ResourceType.AUDIT_RETENTION_DAYS: 365.0
            },
            TenantTier.ENTERPRISE: {
                ResourceType.CPU_CORES: 32.0,
                ResourceType.MEMORY_GB: 64.0,
                ResourceType.STORAGE_GB: 1000.0,
                ResourceType.HEALING_OPERATIONS: 10000.0,
                ResourceType.API_REQUESTS: 100000.0,
                ResourceType.CONCURRENT_JOBS: 50.0,
                ResourceType.USERS: 1000.0,
                ResourceType.AUDIT_RETENTION_DAYS: 2555.0  # 7 years
            }
        }
    
    def initialize_tenant_quotas(self, tenant_id: str, tier: TenantTier):
        """Initialize resource quotas for a new tenant."""
        with self.lock:
            tier_limits = self.tier_limits[tier]
            self.quotas[tenant_id] = {}
            
            for resource_type, limit in tier_limits.items():
                quota = ResourceQuota(
                    resource_type=resource_type,
                    limit=limit,
                    used=0.0,
                    unit=self._get_resource_unit(resource_type)
                )
                self.quotas[tenant_id][resource_type] = quota
            
            logger.info(f"Initialized quotas for tenant {tenant_id} (tier: {tier.value})")
    
    def _get_resource_unit(self, resource_type: ResourceType) -> str:
        """Get unit for resource type."""
        units = {
            ResourceType.CPU_CORES: "cores",
            ResourceType.MEMORY_GB: "GB",
            ResourceType.STORAGE_GB: "GB", 
            ResourceType.HEALING_OPERATIONS: "operations",
            ResourceType.API_REQUESTS: "requests",
            ResourceType.CONCURRENT_JOBS: "jobs",
            ResourceType.USERS: "users",
            ResourceType.AUDIT_RETENTION_DAYS: "days"
        }
        return units.get(resource_type, "units")
    
    def check_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        requested_amount: float = 1.0
    ) -> bool:
        """Check if tenant has quota for requested resource usage."""
        with self.lock:
            if tenant_id not in self.quotas:
                return False
            
            quota = self.quotas[tenant_id].get(resource_type)
            if not quota:
                return False
            
            return quota.available >= requested_amount
    
    def consume_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Consume quota for resource usage."""
        with self.lock:
            if not self.check_quota(tenant_id, resource_type, amount):
                return False
            
            quota = self.quotas[tenant_id][resource_type]
            quota.used += amount
            
            # Record usage
            usage_record = ResourceUsageRecord(
                tenant_id=tenant_id,
                resource_type=resource_type,
                amount=amount,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.usage_records.append(usage_record)
            
            # Clean up old records
            self._cleanup_old_usage_records()
            
            logger.debug(f"Consumed {amount} {quota.unit} of {resource_type.value} for tenant {tenant_id}")
            return True
    
    def release_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: float
    ):
        """Release consumed quota (for concurrent resources)."""
        with self.lock:
            if tenant_id in self.quotas and resource_type in self.quotas[tenant_id]:
                quota = self.quotas[tenant_id][resource_type]
                quota.used = max(0.0, quota.used - amount)
                
                logger.debug(f"Released {amount} {quota.unit} of {resource_type.value} for tenant {tenant_id}")
    
    def get_tenant_quotas(self, tenant_id: str) -> Dict[ResourceType, ResourceQuota]:
        """Get all quotas for a tenant."""
        with self.lock:
            return self.quotas.get(tenant_id, {}).copy()
    
    def get_usage_summary(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage summary for tenant."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Filter usage records
        tenant_records = [
            record for record in self.usage_records
            if (record.tenant_id == tenant_id and
                start_date <= record.timestamp <= end_date)
        ]
        
        # Aggregate usage by resource type
        usage_summary = {}
        for resource_type in ResourceType:
            type_records = [r for r in tenant_records if r.resource_type == resource_type]
            total_usage = sum(r.amount for r in type_records)
            
            usage_summary[resource_type.value] = {
                "total_usage": total_usage,
                "record_count": len(type_records),
                "unit": self._get_resource_unit(resource_type)
            }
        
        return {
            "tenant_id": tenant_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "usage_summary": usage_summary,
            "total_records": len(tenant_records)
        }
    
    def _cleanup_old_usage_records(self):
        """Clean up old usage records to prevent memory leaks."""
        cutoff_date = datetime.now() - timedelta(days=90)
        
        # Keep records from last 90 days
        self.usage_records = [
            record for record in self.usage_records
            if record.timestamp >= cutoff_date
        ]
        
        # Limit total records
        max_records = 100000
        if len(self.usage_records) > max_records:
            self.usage_records = self.usage_records[-max_records:]


class TenantManager:
    """Main tenant management system."""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.resource_manager = ResourceManager()
        self.billing_records: List[BillingRecord] = []
        
        # Tenant isolation contexts
        self.tenant_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Pricing configuration
        self.pricing = self._initialize_pricing()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.billing_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.lock = threading.RLock()
    
    def _initialize_pricing(self) -> Dict[ResourceType, float]:
        """Initialize resource pricing (per unit per month)."""
        return {
            ResourceType.CPU_CORES: 50.0,  # $50/core/month
            ResourceType.MEMORY_GB: 10.0,  # $10/GB/month
            ResourceType.STORAGE_GB: 1.0,  # $1/GB/month
            ResourceType.HEALING_OPERATIONS: 0.01,  # $0.01/operation
            ResourceType.API_REQUESTS: 0.001,  # $0.001/request
            ResourceType.CONCURRENT_JOBS: 20.0,  # $20/concurrent job/month
            ResourceType.USERS: 5.0,  # $5/user/month
            ResourceType.AUDIT_RETENTION_DAYS: 0.1  # $0.1/day/month
        }
    
    async def start(self):
        """Start tenant management services."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.billing_task = asyncio.create_task(self._billing_loop())
        
        logger.info("Multi-tenant manager started")
    
    async def stop(self):
        """Stop tenant management services."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.billing_task:
            self.billing_task.cancel()
        
        logger.info("Multi-tenant manager stopped")
    
    async def create_tenant(
        self,
        name: str,
        tier: TenantTier,
        admin_email: str,
        billing_contact: str,
        compliance_standards: Optional[List[ComplianceStandard]] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> TenantConfig:
        """Create a new tenant."""
        tenant_id = str(uuid.uuid4())
        
        # Set expiration for trial accounts
        expires_at = None
        if tier == TenantTier.BASIC:  # Treat basic as trial
            expires_at = datetime.now() + timedelta(days=30)
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            status=TenantStatus.TRIAL if tier == TenantTier.BASIC else TenantStatus.ACTIVE,
            created_at=datetime.now(),
            admin_email=admin_email,
            billing_contact=billing_contact,
            compliance_standards=compliance_standards or [],
            custom_config=custom_config or {},
            tags=tags or {},
            expires_at=expires_at
        )
        
        with self.lock:
            self.tenants[tenant_id] = tenant_config
            
            # Initialize resource quotas
            self.resource_manager.initialize_tenant_quotas(tenant_id, tier)
            
            # Initialize tenant context
            self.tenant_contexts[tenant_id] = {
                "created_at": datetime.now(),
                "active_sessions": set(),
                "security_config": {},
                "custom_settings": custom_config or {}
            }
        
        # Log compliance event
        await compliance_auditor.log_compliance_event(
            event_type="tenant_created",
            user_id="system",
            resource=f"tenant:{tenant_id}",
            action="create",
            result="success",
            metadata={
                "tenant_name": name,
                "tier": tier.value,
                "admin_email": admin_email
            }
        )
        
        logger.info(f"Created tenant {name} ({tenant_id}) with tier {tier.value}")
        return tenant_config
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        return self.tenants.get(tenant_id)
    
    def authenticate_tenant_context(
        self,
        tenant_id: str,
        security_context: SecurityContext
    ) -> bool:
        """Authenticate and validate tenant context."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        # Check tenant status
        if tenant.status not in [TenantStatus.ACTIVE, TenantStatus.TRIAL]:
            return False
        
        # Check expiration
        if tenant.expires_at and datetime.now() > tenant.expires_at:
            return False
        
        # Add tenant context to security context
        with self.lock:
            if tenant_id in self.tenant_contexts:
                self.tenant_contexts[tenant_id]["active_sessions"].add(security_context.session_id)
        
        return True
    
    async def consume_tenant_resource(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Consume tenant resource with quota checking."""
        # Validate tenant
        tenant = self.get_tenant(tenant_id)
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            return False
        
        # Check quota
        if not self.resource_manager.check_quota(tenant_id, resource_type, amount):
            logger.warning(f"Quota exceeded for tenant {tenant_id}: {resource_type.value}")
            return False
        
        # Consume quota
        success = self.resource_manager.consume_quota(
            tenant_id, resource_type, amount, metadata
        )
        
        if success:
            # Log compliance event for resource usage
            await compliance_auditor.log_compliance_event(
                event_type="resource_consumed",
                user_id=metadata.get("user_id", "system") if metadata else "system",
                resource=f"tenant:{tenant_id}:{resource_type.value}",
                action="consume",
                result="success",
                metadata={
                    "amount": amount,
                    "tenant_tier": tenant.tier.value,
                    **metadata
                } if metadata else {"amount": amount, "tenant_tier": tenant.tier.value}
            )
        
        return success
    
    def release_tenant_resource(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: float = 1.0
    ):
        """Release tenant resource (for concurrent resources)."""
        self.resource_manager.release_quota(tenant_id, resource_type, amount)
    
    def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive tenant status."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}
        
        quotas = self.resource_manager.get_tenant_quotas(tenant_id)
        usage_summary = self.resource_manager.get_usage_summary(tenant_id)
        
        # Calculate days until expiration
        days_until_expiration = None
        if tenant.expires_at:
            days_until_expiration = (tenant.expires_at - datetime.now()).days
        
        # Get active sessions count
        active_sessions = 0
        with self.lock:
            if tenant_id in self.tenant_contexts:
                active_sessions = len(self.tenant_contexts[tenant_id]["active_sessions"])
        
        return {
            "tenant_config": tenant.to_dict(),
            "quotas": {
                resource_type.value: quota.to_dict()
                for resource_type, quota in quotas.items()
            },
            "usage_summary": usage_summary,
            "status": {
                "active_sessions": active_sessions,
                "days_until_expiration": days_until_expiration,
                "billing_status": "current"  # Would integrate with payment system
            }
        }
    
    async def generate_billing_record(
        self,
        tenant_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime
    ) -> BillingRecord:
        """Generate billing record for tenant."""
        usage_summary = self.resource_manager.get_usage_summary(
            tenant_id, billing_period_start, billing_period_end
        )
        
        # Calculate costs
        resource_usage = {}
        total_cost = 0.0
        
        for resource_type_str, usage_data in usage_summary["usage_summary"].items():
            resource_type = ResourceType(resource_type_str)
            usage = usage_data["total_usage"]
            
            # Calculate cost
            unit_price = self.pricing.get(resource_type, 0.0)
            cost = usage * unit_price
            total_cost += cost
            
            resource_usage[resource_type] = usage
        
        billing_record = BillingRecord(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            resource_usage=resource_usage,
            total_cost=total_cost
        )
        
        self.billing_records.append(billing_record)
        
        # Log compliance event
        await compliance_auditor.log_compliance_event(
            event_type="billing_generated",
            user_id="system",
            resource=f"tenant:{tenant_id}",
            action="billing",
            result="success",
            metadata={
                "billing_id": billing_record.id,
                "total_cost": total_cost,
                "currency": billing_record.currency
            }
        )
        
        logger.info(f"Generated billing record for tenant {tenant_id}: ${total_cost:.2f}")
        return billing_record
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                await self._check_tenant_health()
                await self._check_quota_alerts()
                await self._cleanup_expired_sessions()
                
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tenant monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_tenant_health(self):
        """Check health of all tenants."""
        with self.lock:
            tenants_copy = self.tenants.copy()
        
        for tenant_id, tenant in tenants_copy.items():
            # Check expiration
            if tenant.expires_at and datetime.now() > tenant.expires_at:
                if tenant.status == TenantStatus.TRIAL:
                    tenant.status = TenantStatus.EXPIRED
                    logger.info(f"Tenant {tenant_id} trial expired")
            
            # Check quota health
            quotas = self.resource_manager.get_tenant_quotas(tenant_id)
            for resource_type, quota in quotas.items():
                if quota.utilization > 90:
                    logger.warning(
                        f"High resource utilization for tenant {tenant_id}: "
                        f"{resource_type.value} at {quota.utilization:.1f}%"
                    )
    
    async def _check_quota_alerts(self):
        """Check for quota limit alerts."""
        with self.lock:
            tenants_copy = self.tenants.copy()
        
        for tenant_id, tenant in tenants_copy.items():
            quotas = self.resource_manager.get_tenant_quotas(tenant_id)
            
            for resource_type, quota in quotas.items():
                if quota.utilization > 85:
                    # Generate alert
                    enhanced_monitoring.active_alerts[f"quota_{tenant_id}_{resource_type.value}"] = {
                        "tenant_id": tenant_id,
                        "resource_type": resource_type.value,
                        "utilization": quota.utilization,
                        "message": f"High quota utilization: {quota.utilization:.1f}%",
                        "timestamp": datetime.now()
                    }
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired tenant sessions."""
        with self.lock:
            for tenant_id, context in self.tenant_contexts.items():
                # Would integrate with session management to check actual session validity
                # For now, just clean up old session references
                active_sessions = context.get("active_sessions", set())
                
                # In production, would verify sessions with auth manager
                context["active_sessions"] = active_sessions
    
    async def _billing_loop(self):
        """Background billing generation loop."""
        while self.running:
            try:
                # Generate monthly billing on the 1st of each month
                now = datetime.now()
                if now.day == 1 and now.hour == 0:
                    await self._generate_monthly_billing()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in billing loop: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_monthly_billing(self):
        """Generate monthly billing for all tenants."""
        now = datetime.now()
        
        # Calculate previous month period
        if now.month == 1:
            period_start = datetime(now.year - 1, 12, 1)
            period_end = datetime(now.year, now.month, 1) - timedelta(days=1)
        else:
            period_start = datetime(now.year, now.month - 1, 1)
            period_end = datetime(now.year, now.month, 1) - timedelta(days=1)
        
        with self.lock:
            tenants_copy = self.tenants.copy()
        
        for tenant_id, tenant in tenants_copy.items():
            if tenant.status == TenantStatus.ACTIVE:
                try:
                    billing_record = await self.generate_billing_record(
                        tenant_id, period_start, period_end
                    )
                    logger.info(f"Generated monthly billing for tenant {tenant_id}")
                except Exception as e:
                    logger.error(f"Error generating billing for tenant {tenant_id}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get multi-tenant system status."""
        with self.lock:
            tenant_counts = {}
            for tenant in self.tenants.values():
                tier = tenant.tier.value
                status = tenant.status.value
                key = f"{tier}_{status}"
                tenant_counts[key] = tenant_counts.get(key, 0) + 1
            
            total_quotas = {}
            total_usage = {}
            
            for tenant_id in self.tenants.keys():
                quotas = self.resource_manager.get_tenant_quotas(tenant_id)
                for resource_type, quota in quotas.items():
                    resource_key = resource_type.value
                    if resource_key not in total_quotas:
                        total_quotas[resource_key] = 0.0
                        total_usage[resource_key] = 0.0
                    
                    total_quotas[resource_key] += quota.limit
                    total_usage[resource_key] += quota.used
        
        return {
            "total_tenants": len(self.tenants),
            "tenant_breakdown": tenant_counts,
            "resource_totals": {
                "quotas": total_quotas,
                "usage": total_usage,
                "utilization": {
                    resource: (total_usage.get(resource, 0) / total_quotas.get(resource, 1) * 100)
                    for resource in total_quotas.keys()
                }
            },
            "billing_records": len(self.billing_records),
            "last_updated": datetime.now().isoformat()
        }


# Global tenant manager instance
tenant_manager = TenantManager()