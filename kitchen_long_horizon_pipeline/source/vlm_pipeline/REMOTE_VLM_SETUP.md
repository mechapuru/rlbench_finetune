# Remote VLM Setup Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR LOCAL LAPTOP                             │
│  ┌─────────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ RLBench/CoppeliaSim │←→│ VLM Client  │←→│ SSH Tunnel (8000) │   │
│  │  (Visualization)    │    │  (requests) │    │                   │   │
│  └─────────────────┘    └──────────────┘    └─────────┬─────────┘   │
└────────────────────────────────────────────────────────┼─────────────┘
                                                         │ Port 8000
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GVLAB2 GPU SERVER                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    VLM Server (FastAPI)                      │    │
│  │  ┌─────────────────┐    ┌────────────────────────────────┐  │    │
│  │  │ POST /plan      │    │  Qwen2-VL-7B-Instruct (GPU)    │  │    │
│  │  │ GET /health     │→→→→│  or any other VLM model        │  │    │
│  │  └─────────────────┘    └────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Setup

### Step 1: SSH into GPU Server

```bash
# From your laptop terminal
ssh long-horizon@gvlab2.iiit.ac.in -i ~/keyfile
```

### Step 2: Set Up the Code on Server

Once logged in to gvlab2:

```bash
# Clone or copy your codebase to the server
cd ~

# Option A: Clone from git (if you have a repo)
git clone <your-repo-url> TAMP-PDDL

# Option B: Use rsync to copy from your laptop (run this on LAPTOP, not server)
# rsync -avz -e "ssh -i ~/keyfile" /home/naren/iiith/Long_Horizon/TAMP-PDDL long-horizon@gvlab2.iiit.ac.in:~/
```

### Step 3: Install Dependencies on Server

On the server (gvlab2):

```bash
# Create a conda environment (recommended)
conda create -n vlm_server python=3.10 -y
conda activate vlm_server

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install VLM dependencies
pip install transformers accelerate qwen-vl-utils

# Install server dependencies
pip install fastapi uvicorn requests pillow numpy

# For 4-bit quantization (saves GPU memory)
pip install bitsandbytes

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Step 4: Start the VLM Server on gvlab2

On the server:

```bash
cd ~/TAMP-PDDL

# Activate environment
conda activate vlm_server

# Start the server
# Default: Qwen2-VL-7B with 4-bit quantization
python -m vlm_pipeline.vlm_server --port 8000

# Or use the 2B model (lighter, faster)
python -m vlm_pipeline.vlm_server --port 8000 --model Qwen/Qwen2-VL-2B-Instruct --no-4bit

# For background running (keeps running after you disconnect)
nohup python -m vlm_pipeline.vlm_server --port 8000 > vlm_server.log 2>&1 &
```

The server will print:
```
============================================================
VLM INFERENCE SERVER
============================================================
Model: Qwen/Qwen2-VL-7B-Instruct
4-bit quantization: True
Server: http://0.0.0.0:8000
============================================================
[VLMServer] Loading model: Qwen/Qwen2-VL-7B-Instruct
[VLMServer] Model loaded successfully!
```

### Step 5: Create SSH Tunnel from Your Laptop

On your LOCAL laptop (in a new terminal, keep this open):

```bash
# Create SSH tunnel to forward port 8000
ssh -L 8000:localhost:8000 long-horizon@gvlab2.iiit.ac.in -i ~/keyfile -N

# -L 8000:localhost:8000 = Forward local port 8000 to remote port 8000
# -N = Don't execute remote command (just tunnel)
```

**Keep this terminal open!** The tunnel closes if you close the terminal.

### Step 6: Test the Connection

On your laptop, in another terminal:

```bash
cd /home/naren/iiith/Long_Horizon/TAMP-PDDL

# Test the connection
python -m vlm_pipeline.vlm_client

# Or with explicit URL
python -m vlm_pipeline.vlm_client --url http://localhost:8000
```

You should see:
```
============================================================
TESTING VLM SERVER CONNECTION
============================================================
Server URL: http://localhost:8000
Status Code: 200
Server Status: ok
Model Loaded: True
Model Name: Qwen/Qwen2-VL-7B-Instruct
GPU Available: True
============================================================
✓ Connection successful!
```

### Step 7: Use in Your Pipeline

**Option A: Set environment variable**
```bash
export VLM_SERVER_URL="http://localhost:8000"
python -m vlm_pipeline.demo_replanning --record
```

**Option B: Modify code to use RemoteVLMPlanner**

In your code, replace:
```python
from vlm_pipeline.vlm_planner import VLMPlanner
planner = VLMPlanner()
```

With:
```python
from vlm_pipeline.vlm_client import RemoteVLMPlanner
planner = RemoteVLMPlanner(server_url="http://localhost:8000")
```

---

## Quick Reference Commands

### On Your Laptop

```bash
# Terminal 1: SSH Tunnel (keep open)
ssh -L 8000:localhost:8000 long-horizon@gvlab2.iiit.ac.in -i ~/keyfile -N

# Terminal 2: Run your demo
export VLM_SERVER_URL="http://localhost:8000"
python -m vlm_pipeline.demo_replanning --record
```

### On GPU Server (gvlab2)

```bash
# SSH in
ssh long-horizon@gvlab2.iiit.ac.in -i ~/keyfile

# Start server (foreground)
cd ~/TAMP-PDDL && conda activate vlm_server
python -m vlm_pipeline.vlm_server --port 8000

# Start server (background, persists after logout)
nohup python -m vlm_pipeline.vlm_server --port 8000 > vlm_server.log 2>&1 &

# Check if server is running
curl http://localhost:8000/health

# Kill server
pkill -f vlm_server
```

---

## Troubleshooting

### "Cannot connect to server"
1. Check VPN is connected
2. Check SSH tunnel is running
3. Check server is running on gvlab2

### "Model not loaded"
- Check GPU memory on server: `nvidia-smi`
- Try smaller model: `--model Qwen/Qwen2-VL-2B-Instruct`
- Enable 4-bit quantization (default)

### "Request timeout"
- VLM inference takes ~30-60s for first request
- Increase timeout in client if needed

### SSH tunnel dies
- Use `screen` or `tmux` to keep tunnel alive
- Or use autossh: `autossh -L 8000:localhost:8000 ...`

---

## Using Different VLM Models

The server supports any Hugging Face VLM model:

```bash
# Qwen2-VL-7B (default, best quality)
python -m vlm_pipeline.vlm_server --model Qwen/Qwen2-VL-7B-Instruct

# Qwen2-VL-2B (faster, less memory)
python -m vlm_pipeline.vlm_server --model Qwen/Qwen2-VL-2B-Instruct --no-4bit

# LLaVA (if you add support)
python -m vlm_pipeline.vlm_server --model llava-hf/llava-1.5-7b-hf
```

---

## File Sync with Server

To keep code in sync between laptop and server:

```bash
# Sync from laptop to server (run on laptop)
rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='data_states' \
  -e "ssh -i ~/keyfile" \
  /home/naren/iiith/Long_Horizon/TAMP-PDDL/vlm_pipeline/ \
  long-horizon@gvlab2.iiit.ac.in:~/TAMP-PDDL/vlm_pipeline/
```
