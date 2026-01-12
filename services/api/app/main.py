from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.search import router as search_router
from app.routers import recommendations
from app.routers import notifications
from app.routers import chat
from app.routers import documents

app = FastAPI(title="Hedge API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/v1")
app.include_router(recommendations.router)
app.include_router(notifications.router)
app.include_router(chat.router)
app.include_router(documents.router)