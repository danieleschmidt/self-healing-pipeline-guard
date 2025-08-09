#!/usr/bin/env python3
"""
Server entry point for the Self-Healing Pipeline Guard API.

This script launches the FastAPI application with proper configuration.
"""

import asyncio
import logging
import os
import sys
import uvicorn

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from healing_guard.api.main import create_app
from healing_guard.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.monitoring.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        # Create FastAPI app
        app = create_app()
        
        logger.info(f"Starting Healing Guard API server on {settings.host}:{settings.port}")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        
        # Start server
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.monitoring.log_level.lower(),
            access_log=True,
            reload=settings.debug,
            workers=1 if settings.debug else 4
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()