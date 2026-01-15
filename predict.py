"""
Prediction Script Entry Point
This script starts the FastAPI service for serving predictions.
Wraps `src/api/main.py`.
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Production API Service...")
    print("Swagger UI: http://localhost:8000/docs")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
