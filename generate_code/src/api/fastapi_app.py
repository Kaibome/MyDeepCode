"""
FastAPI application instance for the ReAct Agent API.

This module defines the main FastAPI application with CORS middleware,
exception handlers, and router registration for the agent API.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import get_settings

# Configure logger
logger = logging.getLogger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="ReAct Agent API",
    description="A REST API for the ReAct Agent with Tool Integration using LangGraph and DeepSeek",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions and return structured error responses.
    
    Args:
        request: The incoming request.
        exc: The HTTP exception.
        
    Returns:
        JSON response with error details.
    """
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": request.url.path,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle validation errors and return structured error responses.
    
    Args:
        request: The incoming request.
        exc: The validation exception.
        
    Returns:
        JSON response with validation error details.
    """
    logger.error(f"Validation error: {exc.errors()}")
    
    # Format validation errors
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "validation_error"),
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": "Validation failed",
                "path": request.url.path,
                "details": errors,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other exceptions and return structured error responses.
    
    Args:
        request: The incoming request.
        exc: The exception.
        
    Returns:
        JSON response with error details.
    """
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Internal server error",
                "path": request.url.path,
            }
        },
    )


@app.on_event("startup")
async def startup_event() -> None:
    """
    Execute startup tasks.
    
    This function is called when the FastAPI application starts up.
    It can be used to initialize resources, connect to databases, etc.
    """
    logger.info("Starting ReAct Agent API server...")
    
    # Load settings to validate configuration
    settings = get_settings()
    logger.info(f"Loaded settings: {settings.model_dump(exclude={'DEEPSEEK_API_KEY'})}")
    
    # Log server configuration
    server_config = settings.get_server_config()
    logger.info(f"Server will run on {server_config['host']}:{server_config['port']}")
    
    # Log agent configuration
    agent_config = settings.get_agent_config()
    logger.info(f"Agent max iterations: {agent_config['max_iterations']}")
    
    # Import and register routers here to avoid circular imports
    try:
        from src.api.routes import router as api_router
        
        app.include_router(api_router, prefix="/api/v1", tags=["api"])
        logger.info("API routes registered successfully")
    except ImportError as e:
        logger.error(f"Failed to import API routes: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to register API routes: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Execute shutdown tasks.
    
    This function is called when the FastAPI application shuts down.
    It can be used to clean up resources, close connections, etc.
    """
    logger.info("Shutting down ReAct Agent API server...")


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """
    Root endpoint that provides basic API information.
    
    Returns:
        Dictionary with API information and available endpoints.
    """
    return {
        "name": "ReAct Agent API",
        "version": "0.1.0",
        "description": "A REST API for the ReAct Agent with Tool Integration",
        "documentation": "/docs",
        "endpoints": {
            "api": "/api/v1",
            "health": "/api/v1/health",
            "chat": "/api/v1/chat",
            "tools": "/api/v1/tools",
        },
    }


@app.get("/health", include_in_schema=False)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Dictionary with health status and timestamp.
    """
    from datetime import datetime
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "react-agent-api",
        "version": "0.1.0",
    }


def get_app() -> FastAPI:
    """
    Get the FastAPI application instance.
    
    This function is useful for testing and for cases where the app
    needs to be imported without running it directly.
    
    Returns:
        The FastAPI application instance.
    """
    return app


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    server_config = settings.get_server_config()
    
    uvicorn.run(
        "src.api.fastapi_app:app",
        host=server_config["host"],
        port=server_config["port"],
        reload=server_config["reload"],
        log_level="info",
    )