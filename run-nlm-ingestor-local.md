# Running nlm-ingestor Locally

## Option 1: For Local Development (Backend also local)

### Step 1: Run the Docker container locally

```bash
docker run -d \
  --name nlm-ingestor \
  -p 5010:5001 \
  ghcr.io/nlmatics/nlm-ingestor:latest
```

This maps:
- Container port `5001` → Host port `5010`
- Service will be available at `http://localhost:5010`

### Step 2: Update your local backend `.env` file

```env
LLMSHERPA_API_URL=http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true
```

### Step 3: Test locally

```bash
# Check if service is running
curl http://localhost:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true

# Check container logs
docker logs nlm-ingestor
```

---

## Option 2: Expose Local Service to Railway (Using Tunnel)

Since Railway can't access your localhost, you need a tunnel service.

### Step 1: Run Docker container locally

```bash
docker run -d \
  --name nlm-ingestor \
  -p 5010:5001 \
  ghcr.io/nlmatics/nlm-ingestor:latest
```

### Step 2: Install and run ngrok (Recommended)

```bash
# Install ngrok (if not installed)
# macOS: brew install ngrok
# Linux: Download from https://ngrok.com/download
# Windows: Download from https://ngrok.com/download

# Start ngrok tunnel
ngrok http 5010
```

This will give you a public URL like:
```
https://abc123.ngrok.io
```

### Step 3: Update Railway backend environment variable

In Railway, set:
```
LLMSHERPA_API_URL=https://abc123.ngrok.io/api/parseDocument?renderFormat=all&useNewIndentParser=true
```

**Note:** Free ngrok URLs change on restart. For production, consider:
- ngrok paid plan (static domain)
- cloudflared (Cloudflare Tunnel) - free alternative
- localtunnel - free alternative

---

## Option 3: Using cloudflared (Free Alternative)

```bash
# Install cloudflared
# macOS: brew install cloudflared
# Linux: Download from https://github.com/cloudflare/cloudflared/releases

# Run Docker container
docker run -d --name nlm-ingestor -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest

# Start tunnel
cloudflared tunnel --url http://localhost:5010
```

This gives you a public URL like:
```
https://random-subdomain.trycloudflare.com
```

---

## Option 4: Using localtunnel (Free)

```bash
# Install localtunnel
npm install -g localtunnel

# Run Docker container
docker run -d --name nlm-ingestor -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest

# Start tunnel
lt --port 5010
```

This gives you a public URL like:
```
https://random-subdomain.loca.lt
```

---

## Finding Your Local IP Address (For Network Access)

If you want to access from other devices on your network:

```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Or
ip addr show | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig
```

Look for your local network IP (usually `192.168.x.x` or `10.x.x.x`)

Then use: `http://YOUR_LOCAL_IP:5010`

**Note:** This only works if Railway backend is on the same network (not recommended for production).

---

## Docker Commands Reference

```bash
# Start container
docker start nlm-ingestor

# Stop container
docker stop nlm-ingestor

# View logs
docker logs nlm-ingestor
docker logs -f nlm-ingestor  # Follow logs

# Remove container
docker rm nlm-ingestor

# Restart container
docker restart nlm-ingestor
```

---

## Recommended Setup

For **development**: Use Option 1 (everything local)

For **testing with Railway backend**: Use Option 2 with ngrok or cloudflared

For **production**: Keep the service on Railway with increased resources

