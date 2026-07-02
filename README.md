# CapWords Colab GPU Server

Use this for a temporary GPU-backed CapWords server on Google Colab.

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

The server exposes:

- `GET /health`
- `POST /remove-background`
- `POST /object-label`
