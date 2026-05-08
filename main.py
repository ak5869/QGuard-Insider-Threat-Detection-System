from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.db.models import create_tables
from backend.api.routes import auth, uploads, ml, reports

app = FastAPI(title="QuantumWatch API", version="1.0.0",
              docs_url="/api/docs", redoc_url="/api/redoc")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(uploads.router)
app.include_router(ml.router)
app.include_router(reports.router)

static_path = Path(__file__).parent / "frontend" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str = ""):
    index = Path(__file__).parent / "frontend" / "templates" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "QuantumWatch API — visit /api/docs"}


@app.on_event("startup")
def startup():
    create_tables()
    print("⚛️  QuantumWatch started — http://localhost:8000")
    print("📖 API docs — http://localhost:8000/api/docs")
