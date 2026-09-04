from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .engine import MedSAMReconstructionEngine
from .schemas import ReconstructionOptions


engine: MedSAMReconstructionEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    checkpoint = os.environ.get(
        "MEDSAM_INTERACTIVE_CHECKPOINT", "result/best_sdf_sam_slice_24_plane.pth"
    )
    device = os.environ.get("MEDSAM_INTERACTIVE_DEVICE", "cuda")
    engine = MedSAMReconstructionEngine(checkpoint, device)
    yield
    engine = None


app = FastAPI(title="MedSAM implicit reconstruction service", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready" if engine is not None else "starting",
        "device": str(engine.device) if engine is not None else None,
    }


@app.post("/v1/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    options: str = Form(...),
):
    if engine is None:
        raise HTTPException(503, "model is not ready")
    try:
        parsed = ReconstructionOptions.model_validate(json.loads(options))
    except Exception as exc:
        raise HTTPException(422, f"invalid options: {exc}") from exc

    request_dir = Path(tempfile.mkdtemp(prefix="medsam_mitk_"))
    image_path = request_dir / "input.nii.gz"
    mesh_path = request_dir / "reconstruction.ply"
    try:
        with image_path.open("wb") as output:
            while chunk := await image.read(8 * 1024 * 1024):
                output.write(chunk)
        result = engine.reconstruct(image_path, **parsed.model_dump())
        result.mesh.export(mesh_path, file_type="ply")
    except Exception as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(500, str(exc)) from exc

    headers = {
        "X-SDF-Min": str(result.sdf_min),
        "X-SDF-Max": str(result.sdf_max),
        "X-Query-Count": str(result.query_count),
        "X-Grid-Shape": "x".join(map(str, result.grid_shape)),
        "X-Mesh-Coordinates": "image-index",
    }
    return FileResponse(
        mesh_path,
        media_type="application/octet-stream",
        filename="MedSAM_Reconstruction.ply",
        headers=headers,
        background=BackgroundTask(shutil.rmtree, request_dir, ignore_errors=True),
    )
