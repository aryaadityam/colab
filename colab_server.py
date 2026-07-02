import argparse
import ctypes
import glob
import io
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse


def _preload_cuda_libraries() -> None:
    lib_dirs: list[str] = []
    if _torch is not None:
        lib_dirs.append(str(Path(_torch.__file__).resolve().parent / "lib"))

    lib_dirs.extend(
        glob.glob("/usr/local/lib/python*/site-packages/nvidia/*/lib")
    )
    lib_dirs.extend(
        glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")
    )

    existing = [path for path in dict.fromkeys(lib_dirs) if Path(path).is_dir()]
    if existing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            existing + [os.environ.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":")

    for name in (
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
    ):
        for directory in existing:
            candidate = Path(directory) / name
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue


# Import torch first so its CUDA/cuDNN libraries are visible before ONNX Runtime
# initializes the CUDA execution provider on Colab.
try:
    import torch as _torch  # noqa: F401
except Exception:
    _torch = None

_preload_cuda_libraries()

import onnxruntime as ort
from PIL import Image, ImageOps
from rembg import new_session, remove
from starlette.concurrency import run_in_threadpool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CapWords on Colab.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8787, type=int)
    parser.add_argument(
        "--model",
        default="birefnet-general",
        choices=("birefnet-general", "birefnet-general-lite", "u2net"),
    )
    parser.add_argument(
        "--object-label-model",
        default="Salesforce/blip-image-captioning-base",
        help='Set to "off" to disable object labels.',
    )
    parser.add_argument(
        "--tunnel",
        default="cloudflare-quick",
        choices=("none", "cloudflare-quick", "cloudflare-token", "ngrok"),
    )
    parser.add_argument("--cloudflare-token", default="")
    parser.add_argument("--ngrok-token", default="")
    parser.add_argument("--ngrok-domain", default="")
    return parser.parse_args()


ARGS = _parse_args()
os.environ.setdefault("U2NET_HOME", "/content/.u2net")
os.environ["CAPWORDS_SERVER_MODEL"] = ARGS.model
os.environ["CAPWORDS_OBJECT_LABEL_MODEL"] = ARGS.object_label_model
os.environ.setdefault("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024))
os.environ.setdefault("MAX_PROCESS_SIDE", "2048")

MODEL_NAME = os.getenv("CAPWORDS_SERVER_MODEL", "birefnet-general").strip()
OBJECT_LABEL_MODEL = os.getenv(
    "CAPWORDS_OBJECT_LABEL_MODEL",
    "Salesforce/blip-image-captioning-base",
).strip()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_PROCESS_SIDE = int(os.getenv("MAX_PROCESS_SIDE", "2048"))

app = FastAPI(title="CapWords Colab GPU Server")


@app.on_event("startup")
async def warm_up_model() -> None:
    await run_in_threadpool(_model_session)


@app.get("/")
def root() -> dict[str, object]:
    return health()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "provider": "capwords-colab",
        "model": MODEL_NAME,
        "objectLabelModel": OBJECT_LABEL_MODEL if _object_label_enabled() else None,
        "onnxruntimeDevice": ort.get_device(),
        "onnxruntimeProviders": ort.get_available_providers(),
        "maxUploadBytes": MAX_UPLOAD_BYTES,
        "maxProcessSide": MAX_PROCESS_SIDE,
    }


@app.post("/remove-background")
async def remove_background(request: Request) -> Response:
    image_bytes = await _read_limited_body(request)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Body gambar kosong.")

    try:
        png_bytes = await run_in_threadpool(_remove_background_sync, image_bytes)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Server remove background gagal: {error}",
        ) from error

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/object-label")
async def object_label(request: Request) -> JSONResponse:
    image_bytes = await _read_limited_body(request)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Body gambar kosong.")

    try:
        result = await run_in_threadpool(_object_label_sync, image_bytes)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Server object label gagal: {error}",
        ) from error

    return JSONResponse(content=result, headers={"Cache-Control": "no-store"})


async def _read_limited_body(request: Request) -> bytes:
    chunks: list[bytes] = []
    total = 0

    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload terlalu besar. Max {MAX_UPLOAD_BYTES} bytes.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _remove_background_sync(image_bytes: bytes) -> bytes:
    result = remove(_prepare_remove_image(image_bytes), session=_model_session())
    if isinstance(result, bytes):
        return result
    raise RuntimeError("Model tidak mengembalikan PNG bytes.")


def _prepare_remove_image(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    longest_side = max(image.size)
    if MAX_PROCESS_SIDE > 0 and longest_side > MAX_PROCESS_SIDE:
        scale = MAX_PROCESS_SIDE / longest_side
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image.convert("RGBA").save(output, format="PNG")
    else:
        image.convert("RGB").save(output, format="JPEG", quality=95)
    return output.getvalue()


def _object_label_sync(image_bytes: bytes) -> dict[str, object]:
    if not _object_label_enabled():
        return {"label": None, "caption": None, "model": None}

    image = _open_label_image(image_bytes)
    caption = _caption_image(image)
    return {
        "label": _caption_to_label(caption),
        "caption": caption,
        "model": OBJECT_LABEL_MODEL,
    }


def _open_label_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _caption_image(image: Image.Image) -> str:
    processor, model, torch, device = _object_label_session()
    inputs = processor(image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=18)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def _caption_to_label(caption: str) -> str | None:
    text = re.sub(r"\s+", " ", caption.lower()).strip(" .,:;")
    if not text:
        return None

    substitutions = [
        r"^(there is|there are)\s+",
        r"^(a|an|the)\s+(photo|picture|image|photograph)\s+of\s+",
        r"^(a|an|the)\s+close[- ]up\s+of\s+",
        r"^close[- ]up\s+of\s+",
        r"^(a|an|the)\s+pair\s+of\s+",
        r"^(a|an|the)\s+",
    ]
    for pattern in substitutions:
        text = re.sub(pattern, "", text).strip()

    for splitter in (
        " in front of ",
        " next to ",
        " sitting on ",
        " standing on ",
        " lying on ",
        " laying on ",
        " placed on ",
        " with ",
        " on ",
        " in ",
        " at ",
    ):
        if splitter in text:
            text = text.split(splitter, 1)[0].strip()

    text = re.sub(r"[^a-z0-9\s-]", "", text).strip()
    words = [word for word in text.split() if word]
    while words and words[0] in {
        "small",
        "large",
        "big",
        "little",
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "brown",
        "gray",
        "grey",
        "plastic",
        "wooden",
        "metal",
    }:
        words.pop(0)
    words = [
        word
        for word in words
        if word
        not in {
            "object",
            "item",
            "thing",
            "stuff",
            "piece",
            "photo",
            "image",
            "picture",
            "person",
            "someone",
            "hand",
            "hands",
        }
    ]
    if not words:
        return None
    return words[0]


def _object_label_enabled() -> bool:
    return OBJECT_LABEL_MODEL.lower() not in {"", "0", "false", "off", "none"}


@lru_cache(maxsize=1)
def _model_session():
    return new_session(MODEL_NAME)


@lru_cache(maxsize=1)
def _object_label_session():
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(OBJECT_LABEL_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(OBJECT_LABEL_MODEL)
    model.to(device)
    model.eval()
    return processor, model, torch, device


def _install_cloudflared() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing

    target = Path("/usr/local/bin/cloudflared")
    try:
        _download_cloudflared(target)
        return str(target)
    except PermissionError:
        target = Path("/content/cloudflared")
        _download_cloudflared(target)
        return str(target)


def _download_cloudflared(target: Path) -> None:
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64"
    )
    print("Downloading cloudflared...")
    urllib.request.urlretrieve(url, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _start_cloudflare_quick_tunnel(port: int) -> subprocess.Popen:
    cloudflared = _install_cloudflared()
    cmd = [
        cloudflared,
        "tunnel",
        "--url",
        f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_output() -> None:
        for line in process.stdout or []:
            print(line, end="")

    threading.Thread(target=stream_output, daemon=True).start()
    return process


def _start_cloudflare_token_tunnel(token: str) -> subprocess.Popen:
    if not token:
        raise RuntimeError("--cloudflare-token wajib diisi untuk cloudflare-token.")
    cloudflared = _install_cloudflared()
    cmd = [cloudflared, "tunnel", "--no-autoupdate", "run", "--token", token]
    return subprocess.Popen(cmd)


def _start_ngrok_tunnel(port: int, token: str, domain: str) -> object:
    if not token:
        raise RuntimeError("--ngrok-token wajib diisi untuk tunnel ngrok.")
    from pyngrok import ngrok

    ngrok.set_auth_token(token)
    kwargs = {"addr": port, "proto": "http"}
    if domain:
        kwargs["domain"] = domain
    tunnel = ngrok.connect(**kwargs)
    print(f"ngrok public URL: {tunnel.public_url}")
    return tunnel


def _start_tunnel() -> object | None:
    if ARGS.tunnel == "none":
        return None
    if ARGS.tunnel == "cloudflare-quick":
        print("Starting Cloudflare quick tunnel...")
        return _start_cloudflare_quick_tunnel(ARGS.port)
    if ARGS.tunnel == "cloudflare-token":
        print("Starting Cloudflare named tunnel...")
        return _start_cloudflare_token_tunnel(ARGS.cloudflare_token)
    if ARGS.tunnel == "ngrok":
        print("Starting ngrok tunnel...")
        return _start_ngrok_tunnel(ARGS.port, ARGS.ngrok_token, ARGS.ngrok_domain)
    raise RuntimeError(f"Unknown tunnel mode: {ARGS.tunnel}")


def main() -> None:
    print("CapWords Colab server")
    print(f"Model: {MODEL_NAME}")
    print(f"Object label model: {OBJECT_LABEL_MODEL}")
    print(f"ONNX Runtime device before warmup: {ort.get_device()}")
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")
    _start_tunnel()

    import uvicorn

    print(f"Local health: http://127.0.0.1:{ARGS.port}/health")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
