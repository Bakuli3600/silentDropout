from fastapi import FastAPI
from routes import health

def create_app() -> FastAPI:
    """
    Factory function to initialize the FastAPI application.
    """
    app = FastAPI(
        title="Silent Dropout Detection System API",
        description="A modular API for detecting silent dropouts using machine learning.",
        version="1.0.0",
    )

    # Include routers from the routes module
    app.include_router(health.router, tags=["Health"])

    return app

app = create_app()

@app.get("/")
async def root():
    """
    Root endpoint for basic verification.
    """
    return {"message": "Welcome to the Silent Dropout Detection System API"}
