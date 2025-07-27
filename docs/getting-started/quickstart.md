# Quick Start Guide

Get Self-Healing Pipeline Guard up and running in your environment in just 10 minutes! This guide will walk you through the essential steps to start healing your first pipeline failure.

## Prerequisites

Before you begin, ensure you have:

- Docker and Docker Compose installed
- Access to a CI/CD platform (GitHub Actions, GitLab CI, Jenkins, or CircleCI)
- Admin access to configure webhooks
- A modern web browser

## Step 1: Deploy Self-Healing Pipeline Guard

=== "Docker Compose (Recommended)"

    1. **Clone the repository**:
    ```bash
    git clone https://github.com/terragon-labs/self-healing-pipeline-guard.git
    cd self-healing-pipeline-guard
    ```

    2. **Configure environment variables**:
    ```bash
    cp .env.example .env
    # Edit .env with your configuration
    ```

    3. **Start the services**:
    ```bash
    docker-compose up -d
    ```

    4. **Verify deployment**:
    ```bash
    curl http://localhost:8000/health
    # Should return: {"status": "healthy"}
    ```

=== "Docker Run"

    ```bash
    # Start with minimal configuration
    docker run -d \
      --name healing-guard \
      -p 8000:8000 \
      -e DATABASE_URL=sqlite:///healing_guard.db \
      -e REDIS_URL=redis://localhost:6379 \
      terragonlabs/healing-guard:latest
    ```

=== "Kubernetes"

    ```bash
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/
    
    # Wait for deployment
    kubectl wait --for=condition=available deployment/healing-guard
    
    # Get service URL
    kubectl get svc healing-guard
    ```

## Step 2: Configure Your CI/CD Platform

Choose your platform and follow the integration steps:

=== "GitHub Actions"

    1. **Navigate to your repository settings**
    2. **Go to Webhooks section**
    3. **Add a new webhook**:
       - **Payload URL**: `https://your-domain.com/webhooks/github`
       - **Content type**: `application/json`
       - **Secret**: Generate a secure secret
       - **Events**: Select "Workflow runs"

    4. **Add the webhook secret to your environment**:
    ```bash
    echo "GITHUB_WEBHOOK_SECRET=your-secret-here" >> .env
    ```

=== "GitLab CI"

    1. **Go to your project settings**
    2. **Navigate to Webhooks**
    3. **Add webhook**:
       - **URL**: `https://your-domain.com/webhooks/gitlab`
       - **Secret Token**: Generate a secure token
       - **Trigger**: Pipeline events

    4. **Configure the secret**:
    ```bash
    echo "GITLAB_WEBHOOK_SECRET=your-token-here" >> .env
    ```

=== "Jenkins"

    1. **Install the Notification Plugin** (if not already installed)
    2. **Configure global notification**:
       - **Manage Jenkins** → **Configure System**
       - **Job Notifications** → **Add Endpoint**
       - **URL**: `https://your-domain.com/webhooks/jenkins`
       - **Format**: JSON

    3. **Set authentication**:
    ```bash
    echo "JENKINS_TOKEN=your-api-token" >> .env
    ```

## Step 3: Create Your First Healing Strategy

1. **Access the web interface**:
   Open `http://localhost:8000` in your browser

2. **Configure a basic flaky test strategy**:
   ```yaml
   # healing-config.yml
   strategies:
     flaky_test_retry:
       enabled: true
       confidence_threshold: 0.75
       max_retries: 3
       backoff_factor: 2.0
       conditions:
         - failure_rate < 0.5
         - contains_keywords: ["timeout", "connection"]
   ```

3. **Upload the configuration**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/config/strategies \
     -H "Content-Type: application/yaml" \
     --data-binary @healing-config.yml
   ```

## Step 4: Test the Integration

1. **Trigger a test failure** in your pipeline:
   ```yaml
   # Add to your CI workflow for testing
   - name: Test Failure
     run: |
       echo "Simulating flaky test..."
       if [ $((RANDOM % 2)) -eq 0 ]; then
         echo "Test failed due to timeout"
         exit 1
       fi
   ```

2. **Monitor the healing process**:
   - Check the dashboard at `http://localhost:8000/dashboard`
   - View logs: `docker-compose logs -f app`
   - Check metrics: `http://localhost:8000/metrics`

3. **Verify healing worked**:
   ```bash
   # Check recent healing attempts
   curl http://localhost:8000/api/v1/healing/recent
   ```

## Step 5: Monitor and Optimize

1. **Access the monitoring dashboard**:
   - Grafana: `http://localhost:3000` (admin/admin)
   - Prometheus: `http://localhost:9090`

2. **Key metrics to watch**:
   - Healing success rate: Target >80%
   - Detection time: Target <30 seconds
   - False positive rate: Target <5%

3. **Adjust strategies based on results**:
   ```bash
   # Update strategy configuration
   curl -X PUT http://localhost:8000/api/v1/config/strategies/flaky_test_retry \
     -H "Content-Type: application/json" \
     -d '{"confidence_threshold": 0.8}'
   ```

## Verification Checklist

Ensure everything is working correctly:

- [ ] Self-Healing Pipeline Guard is accessible at `http://localhost:8000`
- [ ] Health check returns "healthy" status
- [ ] Webhook endpoint responds to test payload
- [ ] Dashboard shows your connected platform
- [ ] Test failure triggers healing attempt
- [ ] Metrics are being collected
- [ ] Logs show successful webhook processing

## Common Issues and Solutions

??? question "Webhook not receiving events"
    
    **Check these items:**
    - Webhook URL is accessible from the internet
    - Correct webhook secret configured
    - Firewall allows incoming connections on port 8000
    - CI/CD platform has the correct webhook URL

    **Debug steps:**
    ```bash
    # Check webhook logs
    docker-compose logs app | grep webhook
    
    # Test webhook endpoint manually
    curl -X POST http://localhost:8000/webhooks/github \
      -H "Content-Type: application/json" \
      -d '{"test": "payload"}'
    ```

??? question "Healing strategies not triggering"
    
    **Possible causes:**
    - Strategy configuration incorrect
    - Confidence threshold too high
    - Failure pattern doesn't match conditions
    - Insufficient permissions for healing actions

    **Debug steps:**
    ```bash
    # Check strategy status
    curl http://localhost:8000/api/v1/strategies/status
    
    # Review failure analysis
    curl http://localhost:8000/api/v1/failures/recent
    ```

??? question "High memory usage"
    
    **Solutions:**
    - Reduce ML model memory usage
    - Increase Docker memory limits
    - Enable log rotation
    - Tune garbage collection

    **Configuration:**
    ```yaml
    # docker-compose.override.yml
    services:
      app:
        deploy:
          resources:
            limits:
              memory: 2G
            reservations:
              memory: 1G
    ```

## Next Steps

Now that you have Self-Healing Pipeline Guard running:

1. **Explore advanced strategies**: [Healing Strategies Guide](../user-guide/strategies/overview.md)
2. **Set up comprehensive monitoring**: [Monitoring Guide](../user-guide/monitoring.md)
3. **Configure additional platforms**: [Platform Integration](../user-guide/platforms/)
4. **Optimize for your environment**: [Configuration Reference](../reference/configuration.md)
5. **Learn about custom strategies**: [Custom Strategies](../user-guide/strategies/custom.md)

## Getting Help

Need assistance? Here are your options:

- **Documentation**: Browse the complete [User Guide](../user-guide/)
- **API Reference**: Detailed [API documentation](../api/)
- **Community**: Join our [GitHub Discussions](https://github.com/terragon-labs/self-healing-pipeline-guard/discussions)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/terragon-labs/self-healing-pipeline-guard/issues)
- **Commercial Support**: Contact sales@terragonlabs.com

---

🎉 **Congratulations!** You've successfully set up Self-Healing Pipeline Guard. Your pipelines are now protected by intelligent automation that will reduce failures and save valuable development time.