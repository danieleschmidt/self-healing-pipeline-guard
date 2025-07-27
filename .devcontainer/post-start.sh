#!/bin/bash

# Post-start script for development container
set -e

echo "🔄 Starting development services..."

# Wait for services to be ready
echo "⏳ Waiting for database and cache services..."
python scripts/wait_for_services.py

# Run database migrations
echo "🗄️ Running database migrations..."
alembic upgrade head

# Start background services for development
echo "🔧 Starting background services..."
# Note: In development, these might be started by docker-compose

# Display service status
echo "📊 Service status:"
python scripts/health_check.py

echo "✅ Development environment ready!"
echo ""
echo "Services available:"
echo "  API Server: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Health Dashboard: http://localhost:8080"
echo ""