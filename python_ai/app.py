from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from api.endpoints import router  # semua endpoints disatukan di sini

# Inisialisasi FastAPI
app = FastAPI(
    title="Advanced Stock Prediction API",
    description="API untuk prediksi saham dengan LightGBM, FinBERT, dan LLM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bisa diganti domain production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include router dari folder api/endpoints.py
app.include_router(router)

# Endpoint root
@app.get("/")
async def root():
    return {"message": "Stock Prediction API is running!"}

# Jika mau run langsung: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
