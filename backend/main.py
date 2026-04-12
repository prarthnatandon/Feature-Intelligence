"""
FastAPI backend — SSE streaming, pipeline orchestration, static file serving.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agents import run_pipeline
from backend.models import CompanyContext, FeatureBrief, FeedbackBundle, SSEEvent
from backend.scraper import fetch_bundle

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Duolingo Feature Intelligence", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

BRIEFS_DIR = Path(__file__).parent.parent / "briefs"
BRIEFS_DIR.mkdir(exist_ok=True)

# In-memory run store
runs: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    company_name: str = "Duolingo"
    product_description: str = "language learning app"
    subreddit: str = ""
    app_store_id: str = ""
    known_features: str = ""


class AnalyzeResponse(BaseModel):
    run_id: str
    company_name: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    path = FRONTEND_DIR / "index.html"
    if path.exists():
        return FileResponse(str(path))
    return {"message": "Duolingo Feature Intelligence API — set ANTHROPIC_API_KEY and open /docs"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def start_analysis(request: AnalyzeRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in environment")

    company = CompanyContext(
        company_name=request.company_name.strip() or "Duolingo",
        product_description=request.product_description.strip() or "consumer app",
        subreddit=request.subreddit.strip().lstrip("r/"),
        app_store_id=request.app_store_id.strip(),
        known_features=request.known_features.strip(),
    )

    run_id = str(uuid.uuid4())
    queue: asyncio.Queue[Optional[SSEEvent]] = asyncio.Queue()

    runs[run_id] = {
        "queue": queue,
        "brief": None,
        "status": "starting",
        "company": company,
    }

    asyncio.create_task(_pipeline_task(run_id, api_key, company))
    return AnalyzeResponse(run_id=run_id, company_name=company.company_name)


@app.get("/stream/{run_id}")
async def stream_events(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    queue = runs[run_id]["queue"]

    async def generator():
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id})}\n\n"
        while True:
            try:
                event: Optional[SSEEvent] = await asyncio.wait_for(
                    queue.get(), timeout=120.0
                )
            except asyncio.TimeoutError:
                yield "data: " + json.dumps({"type": "keepalive"}) + "\n\n"
                continue

            if event is None:
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                break

            yield "data: " + event.model_dump_json() + "\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/brief/{run_id}")
async def get_brief(run_id: str):
    run = runs.get(run_id)
    if run and run.get("brief"):
        return run["brief"].model_dump(mode="json")
    # Try loading from disk
    brief_path = BRIEFS_DIR / f"{run_id}.json"
    if brief_path.exists():
        return json.loads(brief_path.read_text())
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    raise HTTPException(status_code=425, detail="Brief not ready yet")


@app.get("/brief/{run_id}/markdown")
async def get_brief_markdown(run_id: str):
    run = runs.get(run_id)
    if not run or not run.get("brief"):
        raise HTTPException(status_code=425, detail="Brief not ready yet")
    return {"markdown": run["brief"].full_markdown}


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": run["status"]}


# ---------------------------------------------------------------------------
# Background pipeline task
# ---------------------------------------------------------------------------

async def _pipeline_task(run_id: str, api_key: str, company: CompanyContext):
    queue = runs[run_id]["queue"]
    client = anthropic.AsyncAnthropic(api_key=api_key)

    try:
        # Phase 1: Fetch data
        runs[run_id]["status"] = "fetching"
        await queue.put(SSEEvent(
            type="phase_start", phase="fetching",
            message=f"Fetching {company.company_name} feedback from Reddit and App Store...",
        ))

        async def progress_cb(msg: str):
            await queue.put(SSEEvent(type="progress", message=msg))

        bundle: FeedbackBundle = await fetch_bundle(company=company, progress_cb=progress_cb)

        await queue.put(SSEEvent(
            type="phase_complete", phase="fetching",
            message=f"Loaded {bundle.total} {company.company_name} feedback items",
            data={
                "total": bundle.total,
                "reddit": bundle.reddit_count,
                "app_store": bundle.app_store_count,
                "google_play": bundle.google_play_count,
                "hacker_news": bundle.hacker_news_count,
                "seed": bundle.seed_count,
            },
        ))

        # Phase 2: Agent pipeline
        runs[run_id]["status"] = "analyzing"
        brief = await run_pipeline(client, bundle, company, queue)
        runs[run_id]["brief"] = brief
        runs[run_id]["status"] = "complete"

        # Persist to disk
        try:
            brief_path = BRIEFS_DIR / f"{run_id}.json"
            brief_path.write_text(brief.model_dump_json())
        except Exception as e:
            logger.warning(f"Failed to persist brief {run_id}: {e}")

        await queue.put(SSEEvent(
            type="phase_complete", phase="pipeline",
            message="Analysis complete.",
        ))

    except Exception as e:
        logger.exception(f"Pipeline failed for run {run_id}")
        runs[run_id]["status"] = "error"
        await queue.put(SSEEvent(type="error", error=str(e)))
    finally:
        await queue.put(None)  # sentinel — closes SSE stream
