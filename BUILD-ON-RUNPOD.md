# Backup: Build Docker Image on RunPod Pod

If GitHub Actions fails (disk space, timeout), build directly on a RunPod pod.

## 1. Start a cheap CPU pod on RunPod

- Go to https://www.runpod.io/console/pods
- Deploy a **CPU pod** (cheapest option, ~$0.10/hr)
- Template: **RunPod Ubuntu** or any Linux pod
- Disk: **50 GB** minimum
- Start the pod and open a terminal

## 2. Install Docker (if not pre-installed)

```bash
curl -fsSL https://get.docker.com | sh
```

## 3. Clone, build, and push

```bash
# Clone the repo
git clone https://github.com/lovelivemusic/whisperx-worker.git
cd whisperx-worker

# Login to Docker Hub
docker login -u wearethemods

# Build (native x86, no cross-compilation needed)
# Pass HF token as build secret for model downloads
echo "$HF_TOKEN" > /tmp/hf_token
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=/tmp/hf_token -t wearethemods/whisperx-worker:latest .
rm /tmp/hf_token

# Push
docker push wearethemods/whisperx-worker:latest
```

## 4. Cleanup

- Stop and delete the pod
- Delete all workers on your serverless endpoint so new ones pull the fresh image
