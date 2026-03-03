from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import health, prediction

def create_app() -> FastAPI:
    """
    Factory function to initialize the FastAPI application.
    """
    app = FastAPI(
        title="Silent Dropout Detection System API",
        description="A modular API for detecting silent dropouts using machine learning.",
        version="1.0.0",
    )

    # Enable CORS for the frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers from the routes module
    app.include_router(health.router, tags=["Health"])
    app.include_router(prediction.router, tags=["Prediction"])

    return app

app = create_app()

@app.get("/")
async def root():
    """
    Root endpoint for basic verification.
    """
    return {"message": "Welcome to the Silent Dropout Detection System API"}
