from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import uuid, json

# import your engine
from bifurcation import QMREngine  # adjust if name differs

ALLOWED_ORIGINS = ["https://proofx.org", "https://www.proofx.org", "https://proof-x.vercel.app"]

app = FastAPI(title="ProofX Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
API_KEY = "CHANGE_ME"  # set via env var in real prod

def new_engine(cfg: dict | None = None) -> QMREngine:
    base = {"mode": "HYBRID", "universe": "EUCLIDEAN"}
    if cfg: base.update(cfg)
    return QMREngine(base)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/jobs/explore")
def explore(background: BackgroundTasks, depth: int = 1, x_api_key: str = Header(None)):
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(401, "Unauthorized")
    job_id = str(uuid.uuid4())
    out = ART / f"{job_id}.json"

    def run():
        eng = new_engine()
        result = eng.explore(depth=depth, base=[])
        out.write_text(json.dumps({"depth": depth, "result": str(result)}, indent=2))

    background.add_task(run)
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
def status(job_id: str):
    f = ART / f"{job_id}.json"
    return {"job_id": job_id, "ready": f.exists()}

@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    f = ART / f"{job_id}.json"
    if not f.exists(): raise HTTPException(404, "Not ready")
    return FileResponse(f, media_type="application/json")

@app.get("/")
def root(): return JSONResponse({"service":"ProofX Engine", "docs":"/docs"})
