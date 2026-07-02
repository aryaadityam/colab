import argparse
import io
import os
import shutil
import stat
import subprocess
import sys
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

import onnxruntime as ort
from PIL import Image, ImageColor, ImageOps
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
        default="off",
        help=argparse.SUPPRESS,
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
os.environ.setdefault("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024))
os.environ.setdefault("MAX_PROCESS_SIDE", "2048")
os.environ.setdefault("MAX_VIDEO_UPLOAD_BYTES", str(80 * 1024 * 1024))
os.environ.setdefault("MAX_VIDEO_SECONDS", "25")
os.environ.setdefault("MAX_VIDEO_FPS", "30")
os.environ.setdefault("MAX_VIDEO_SIDE", "720")
os.environ.setdefault("RVM_MODEL", "mobilenetv3")
os.environ.setdefault("RVM_DOWNSAMPLE_RATIO", "auto")
os.environ.setdefault("CAPWORDS_MODEL_HOME", "/content/.cache/capwords")
os.environ.setdefault("PRELOAD_VIDEO_MODEL", "1")

MODEL_NAME = os.getenv("CAPWORDS_SERVER_MODEL", "birefnet-general").strip()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_PROCESS_SIDE = int(os.getenv("MAX_PROCESS_SIDE", "2048"))
MAX_VIDEO_UPLOAD_BYTES = int(
    os.getenv("MAX_VIDEO_UPLOAD_BYTES", str(80 * 1024 * 1024))
)
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "25"))
MAX_VIDEO_FPS = int(os.getenv("MAX_VIDEO_FPS", "30"))
MAX_VIDEO_SIDE = int(os.getenv("MAX_VIDEO_SIDE", "720"))
RVM_MODEL = os.getenv("RVM_MODEL", "mobilenetv3").strip() or "mobilenetv3"
RVM_DOWNSAMPLE_RATIO = os.getenv("RVM_DOWNSAMPLE_RATIO", "auto").strip()
MODEL_HOME = Path(os.getenv("CAPWORDS_MODEL_HOME", "/content/.cache/capwords"))
PRELOAD_VIDEO_MODEL = os.getenv("PRELOAD_VIDEO_MODEL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RVM_TORCHSCRIPT_URLS = {
    "mobilenetv3": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
        "v1.0.0/rvm_mobilenetv3_fp32.torchscript"
    ),
    "resnet50": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
        "v1.0.0/rvm_resnet50_fp32.torchscript"
    ),
}

app = FastAPI(title="CapWords Colab GPU Server")


@app.on_event("startup")
async def warm_up_model() -> None:
    print(f"Preloading image model: {MODEL_NAME}")
    await run_in_threadpool(_model_session)
    if PRELOAD_VIDEO_MODEL:
        print(f"Preloading video model: RVM {RVM_MODEL}")
        await run_in_threadpool(_rvm_session)
    print("Model preload complete.")


@app.get("/")
def root() -> dict[str, object]:
    return health()


@lru_cache(maxsize=1)
def _torch_cuda_status() -> dict[str, object]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        return {
            "torchCudaAvailable": cuda_available,
            "torchCudaVersion": getattr(torch.version, "cuda", None),
            "torchDevice": torch.cuda.get_device_name(0)
            if cuda_available
            else None,
        }
    except Exception as error:
        return {
            "torchCudaAvailable": False,
            "torchCudaVersion": None,
            "torchDevice": None,
            "torchImportError": str(error),
        }


@app.get("/health")
def health() -> dict[str, object]:
    torch_status = _torch_cuda_status()
    return {
        "ok": True,
        "provider": "capwords-colab",
        "model": MODEL_NAME,
        "objectLabelModel": None,
        "onnxruntimeDevice": ort.get_device(),
        "onnxruntimeProviders": ort.get_available_providers(),
        "maxUploadBytes": MAX_UPLOAD_BYTES,
        "maxProcessSide": MAX_PROCESS_SIDE,
        "maxVideoUploadBytes": MAX_VIDEO_UPLOAD_BYTES,
        "maxVideoSeconds": MAX_VIDEO_SECONDS,
        "maxVideoFps": MAX_VIDEO_FPS,
        "maxVideoSide": MAX_VIDEO_SIDE,
        "videoModel": f"rvm-{RVM_MODEL}",
        "videoDownsampleRatio": RVM_DOWNSAMPLE_RATIO,
        "preloadVideoModel": PRELOAD_VIDEO_MODEL,
        **torch_status,
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


@app.post("/remove-video-background")
async def remove_video_background(request: Request) -> Response:
    video_bytes = await _read_limited_body(
        request,
        max_bytes=MAX_VIDEO_UPLOAD_BYTES,
    )
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Body video kosong.")

    output_format = request.query_params.get("format", "webm")
    background = request.query_params.get("background", "white")

    try:
        media_type, output_bytes = await run_in_threadpool(
            _remove_video_background_sync,
            video_bytes,
            output_format,
            background,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Server remove video background gagal: {error}",
        ) from error

    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/object-label")
async def object_label(request: Request) -> JSONResponse:
    await _read_limited_body(request)
    return JSONResponse(
        content={"label": None, "caption": None, "model": None},
        headers={"Cache-Control": "no-store"},
    )


async def _read_limited_body(
    request: Request,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> bytes:
    chunks: list[bytes] = []
    total = 0

    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload terlalu besar. Max {max_bytes} bytes.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _remove_background_sync(image_bytes: bytes, max_side: int | None = None) -> bytes:
    result = remove(
        _prepare_remove_image(image_bytes, max_side=max_side),
        session=_model_session(),
    )
    if isinstance(result, bytes):
        return result
    raise RuntimeError("Model tidak mengembalikan PNG bytes.")


def _prepare_remove_image(image_bytes: bytes, max_side: int | None = None) -> bytes:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    longest_side = max(image.size)
    process_side = MAX_PROCESS_SIDE if max_side is None else max_side
    if process_side > 0 and longest_side > process_side:
        scale = process_side / longest_side
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


def _remove_video_background_sync(
    video_bytes: bytes,
    output_format: str,
    background: str,
) -> tuple[str, bytes]:
    normalized_format = _normalize_video_format(output_format)
    ffmpeg = _ffmpeg_bin()

    with TemporaryDirectory(prefix="capwords-video-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input-video"
        source_frames = tmp_path / "source-frames"
        output_frames = tmp_path / "output-frames"
        source_frames.mkdir()
        output_frames.mkdir()
        input_path.write_bytes(video_bytes)

        _run_command(_extract_frames_command(ffmpeg, input_path, source_frames))
        frames = sorted(source_frames.glob("frame_*.png"))
        if not frames:
            raise RuntimeError("Tidak ada frame video yang bisa diproses.")

        _write_rvm_video_frames(
            frames,
            output_frames,
            output_format=normalized_format,
            background=background,
        )

        if normalized_format == "mp4":
            output_path = tmp_path / "output.mp4"
            _run_command(_encode_mp4_command(ffmpeg, output_frames, output_path))
            return "video/mp4", output_path.read_bytes()

        output_path = tmp_path / "output.webm"
        _run_command(_encode_webm_command(ffmpeg, output_frames, output_path))
        return "video/webm", output_path.read_bytes()


def _normalize_video_format(output_format: str) -> str:
    value = output_format.strip().lower()
    if value in {"", "webm", "transparent", "alpha"}:
        return "webm"
    if value in {"mp4", "mpeg4"}:
        return "mp4"
    raise ValueError("Format video harus webm atau mp4.")


def _extract_frames_command(
    ffmpeg: str,
    input_path: Path,
    frames_dir: Path,
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if MAX_VIDEO_SECONDS > 0:
        command.extend(["-t", _format_number(MAX_VIDEO_SECONDS)])
    command.extend(["-i", str(input_path), "-an"])
    filter_graph = _video_filter_graph()
    if filter_graph:
        command.extend(["-vf", filter_graph])
    command.append(str(frames_dir / "frame_%06d.png"))
    return command


def _video_filter_graph() -> str:
    filters: list[str] = []
    fps = _video_fps()
    if fps > 0:
        filters.append(f"fps={fps}")
    if MAX_VIDEO_SIDE > 0:
        filters.append(
            "scale="
            f"'if(gte(a,1),min(iw,{MAX_VIDEO_SIDE}),-2)':"
            f"'if(gte(a,1),-2,min(ih,{MAX_VIDEO_SIDE}))':"
            "flags=lanczos"
        )
    filters.append("setsar=1")
    return ",".join(filters)


def _encode_webm_command(
    ffmpeg: str,
    frames_dir: Path,
    output_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(_video_fps()),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-row-mt",
        "1",
        "-b:v",
        "0",
        "-crf",
        "32",
        str(output_path),
    ]


def _encode_mp4_command(
    ffmpeg: str,
    frames_dir: Path,
    output_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(_video_fps()),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "medium",
        "-crf",
        "20",
        str(output_path),
    ]


def _write_rvm_video_frames(
    frames: list[Path],
    output_frames: Path,
    *,
    output_format: str,
    background: str,
) -> None:
    model, torch, device = _rvm_session()
    downsample_ratio = _rvm_downsample_ratio()
    rec = [None, None, None, None]

    with torch.inference_mode():
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            source = _pil_rgb_to_torch(image, torch, device)
            image.close()

            foreground, alpha, *rec = model(source, *rec, downsample_ratio)
            rgba = _rvm_output_to_rgba(foreground, alpha, torch)
            if output_format == "mp4":
                rgba = _flatten_rgba(rgba, background)
            rgba.save(output_frames / frame.name, format="PNG")
            rgba.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pil_rgb_to_torch(image: Image.Image, torch: object, device: str) -> object:
    import numpy as np

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device, non_blocking=True)


def _rvm_output_to_rgba(foreground: object, alpha: object, torch: object) -> Image.Image:
    import numpy as np

    foreground = foreground[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    alpha = alpha[0, 0].detach().clamp(0, 1).cpu().numpy()
    rgba = np.empty((foreground.shape[0], foreground.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = (foreground * 255.0).round().astype(np.uint8)
    rgba[:, :, 3] = (alpha * 255.0).round().astype(np.uint8)
    del foreground, alpha
    return Image.fromarray(rgba, mode="RGBA")


def _rvm_downsample_ratio() -> float:
    value = RVM_DOWNSAMPLE_RATIO.lower()
    if value in {"", "auto", "none", "default"}:
        if MAX_VIDEO_SIDE <= 720:
            return 0.375
        if MAX_VIDEO_SIDE <= 1080:
            return 0.25
        return 0.2
    try:
        ratio = float(value)
    except ValueError as error:
        raise RuntimeError("RVM_DOWNSAMPLE_RATIO harus auto atau angka float.") from error
    if ratio <= 0:
        return 0.375
    return ratio


def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    message = (result.stderr or result.stdout or "ffmpeg gagal.").strip()
    raise RuntimeError(message[-1400:])


def _flatten_rgba(image: Image.Image, background: str) -> Image.Image:
    rgb = _background_rgb(background)
    canvas = Image.new("RGBA", image.size, (*rgb, 255))
    canvas.alpha_composite(image)
    return canvas.convert("RGB")


def _background_rgb(background: str) -> tuple[int, int, int]:
    presets = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (230, 230, 222),
        "grey": (230, 230, 222),
        "green": (0, 255, 0),
    }
    value = background.strip().lower()
    if value in presets:
        return presets[value]
    try:
        color = ImageColor.getrgb(value)
    except ValueError:
        return presets["white"]
    return color[:3]


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _video_fps() -> int:
    return max(1, MAX_VIDEO_FPS)


@lru_cache(maxsize=1)
def _ffmpeg_bin() -> str:
    existing = shutil.which("ffmpeg")
    if existing:
        return existing
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "ffmpeg belum tersedia. Install ffmpeg atau imageio-ffmpeg."
        ) from error


@lru_cache(maxsize=1)
def _model_session():
    return new_session(MODEL_NAME)


@lru_cache(maxsize=1)
def _rvm_session() -> tuple[object, object, str]:
    import torch

    if RVM_MODEL not in {"mobilenetv3", "resnet50"}:
        raise RuntimeError("RVM_MODEL harus mobilenetv3 atau resnet50.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = _download_rvm_torchscript(RVM_MODEL)
    model = torch.jit.load(str(model_path), map_location=device).eval()
    try:
        model = torch.jit.freeze(model)
    except Exception:
        pass
    return model, torch, device


def _download_rvm_torchscript(model_name: str) -> Path:
    url = RVM_TORCHSCRIPT_URLS[model_name]
    MODEL_HOME.mkdir(parents=True, exist_ok=True)
    target = MODEL_HOME / f"rvm_{model_name}_fp32.torchscript"
    if target.exists() and target.stat().st_size > 0:
        return target

    temp_target = target.with_suffix(".download")
    print(f"Downloading RVM TorchScript model: {url}")
    urllib.request.urlretrieve(url, temp_target)
    temp_target.replace(target)
    return target


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
    print("Object label model: disabled")
    print(
        "Video: "
        f"RVM {RVM_MODEL}, max {MAX_VIDEO_SECONDS}s, "
        f"{MAX_VIDEO_FPS} fps, {MAX_VIDEO_SIDE}px side"
    )
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
