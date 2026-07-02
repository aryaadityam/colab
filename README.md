# CapWords Colab GPU Server

Use this for a temporary GPU-backed CapWords server on Google Colab.

Open directly in Colab:

```text
https://colab.research.google.com/github/aryaadityam/colab/blob/main/CapWords_Colab_Server.ipynb
```

Do not open `colab_server.py` as a notebook. It is a Python script, not an
`.ipynb` JSON notebook.

## Colab Setup

1. Open a Colab notebook and set runtime to GPU.
2. Upload `colab_server.py` and `requirements-colab.txt`.
3. Run:

```bash
!pip install -q -r requirements-colab.txt
```

## Domain Option: Cloudflare Tunnel

Recommended for a personal domain. Put your domain on Cloudflare, create a
Cloudflare Tunnel in Zero Trust, map the public hostname to
`http://localhost:8787`, then copy the tunnel token.

Run:

```bash
!python colab_server.py \
  --model birefnet-general \
  --tunnel cloudflare-token \
  --cloudflare-token "PASTE_CLOUDFLARE_TUNNEL_TOKEN"
```

Your app endpoint becomes:

```text
https://your-subdomain.your-domain.com
```

## Temporary URL Option: Cloudflare Quick Tunnel

No account setup, but the URL changes every runtime.

```bash
!python colab_server.py --model birefnet-general --tunnel cloudflare-quick
```

## Temporary URL Option: ngrok

```bash
!python colab_server.py \
  --model birefnet-general \
  --tunnel ngrok \
  --ngrok-token "PASTE_NGROK_TOKEN"
```

If your ngrok account supports a reserved/custom domain:

```bash
!python colab_server.py \
  --model birefnet-general \
  --tunnel ngrok \
  --ngrok-token "PASTE_NGROK_TOKEN" \
  --ngrok-domain "api.your-domain.com"
```

For this project endpoint:

```bash
!python colab_server.py \
  --model birefnet-general \
  --tunnel ngrok \
  --ngrok-token "PASTE_NGROK_TOKEN" \
  --ngrok-domain "squeak-barracuda-clench.ngrok-free.dev"
```

## Models

- `birefnet-general`: best quality, heavier.
- `birefnet-general-lite`: faster/lighter.
- `u2net`: stable fallback.

Image background removal uses the selected rembg model above. Video background
removal always uses Robust Video Matting (`RVM_MODEL=mobilenetv3` by default)
inside the same server.

The server exposes:

- `GET /health`
- `POST /remove-background`
- `POST /remove-video-background`
- `POST /object-label` returns an empty disabled label response for old app
  compatibility.

Oversized input images are resized to `MAX_PROCESS_SIDE=2048` on the longest
side before background removal.

Video defaults are intentionally small for Colab demos:

- `MAX_VIDEO_UPLOAD_BYTES=83886080`
- `MAX_VIDEO_SECONDS=6`
- `MAX_VIDEO_FPS=12`
- `MAX_VIDEO_SIDE=960`
- `RVM_MODEL=mobilenetv3`
- `RVM_DOWNSAMPLE_RATIO=auto`

Default video output is transparent WebM:

```bash
curl -X POST --data-binary @input.mp4 \
  "https://your-server/remove-video-background?format=webm" \
  -o output.webm
```

For MP4 compatibility, use a flattened background:

```bash
curl -X POST --data-binary @input.mp4 \
  "https://your-server/remove-video-background?format=mp4&background=white" \
  -o output.mp4
```
