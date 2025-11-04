import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import inspect
from typing import List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from modelscope import AutoModel
from modelscope.hub.snapshot_download import snapshot_download

MODEL_ID = "iic/gme-Qwen2-VL-2B-Instruct"
MODEL_ROOT = os.environ.get(
    "GME_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), "gme_models"),
)

torch.backends.cudnn.benchmark = True

app = FastAPI(title="GME-Qwen2-VL-2B Embedding Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model... (首次会下载权重)")

os.makedirs(MODEL_ROOT, exist_ok=True)
snapshot_download_signature = inspect.signature(snapshot_download)
snapshot_kwargs = {}
if "trust_remote_code" in snapshot_download_signature.parameters:
    snapshot_kwargs["trust_remote_code"] = True

model_local_path = snapshot_download(
    MODEL_ID,
    cache_dir=MODEL_ROOT,
    **snapshot_kwargs,
)

auto_model_signature = inspect.signature(AutoModel.from_pretrained)
auto_model_kwargs = {
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "trust_remote_code": True,
}
filtered_auto_model_kwargs = {
    key: value
    for key, value in auto_model_kwargs.items()
    if key in auto_model_signature.parameters
}

gme = AutoModel.from_pretrained(
    model_local_path,
    **filtered_auto_model_kwargs,
)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel inference.")
    gme = torch.nn.DataParallel(gme)

gme.eval()


def _get_model():
    if isinstance(gme, torch.nn.DataParallel):
        return gme.module
    return gme


class TextRequest(BaseModel):
    texts: List[str]
    prompt: Optional[str] = None

@torch.no_grad()
def encode_text(texts: List[str], prompt: Optional[str] = None):
    model = _get_model()
    if prompt:
        emb = model.get_text_embeddings(texts=texts, instruction=prompt)
    else:
        emb = model.get_text_embeddings(texts=texts)
    return emb.detach().float().cpu().tolist()

@torch.no_grad()
def encode_image(images: List[Image.Image]):
    pil_images = [img.convert("RGB") for img in images]
    model = _get_model()
    emb = model.get_image_embeddings(images=pil_images)
    return emb.detach().float().cpu().tolist()

@app.post("/embed/text")
def embed_text(payload: TextRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts 不能为空")
    return {"embeddings": encode_text(payload.texts, payload.prompt)}

@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    try:
        buf = await file.read()
        image = Image.open(io.BytesIO(buf))
    except Exception:
        raise HTTPException(status_code=400, detail="无法解码图片")
    return {"embeddings": encode_image([image])}
