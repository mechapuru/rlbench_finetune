#!/bin/bash
# =============================================================================
# sync_to_server.sh - Sync vlm_pipeline code to GPU server
# =============================================================================
# Usage: ./sync_to_server.sh
# =============================================================================

SERVER="long-horizon@gvlab2.iiit.ac.in"
KEYFILE="$HOME/.ssh/id_ed25519"
LOCAL_DIR="/home/naren/iiith/Long_Horizon/TAMP-PDDL"
REMOTE_DIR="~/TAMP-PDDL"

echo "=============================================="
echo "   SYNC CODE TO GPU SERVER"
echo "=============================================="
echo "Server: $SERVER"
echo "Local:  $LOCAL_DIR"
echo "Remote: $REMOTE_DIR"
echo "=============================================="

# Check if keyfile exists
if [ ! -f "$KEYFILE" ]; then
    echo "ERROR: Keyfile not found at $KEYFILE"
    exit 1
fi

# Create remote directories first
echo ""
echo "Creating remote directories..."
ssh -i "$KEYFILE" "$SERVER" "mkdir -p $REMOTE_DIR/vlm_pipeline $REMOTE_DIR/pddl"

# Sync vlm_pipeline folder (using scp)
echo ""
echo "Syncing vlm_pipeline/..."
# Copy only .py files (avoids __pycache__, .pyc, etc.)
scp -i "$KEYFILE" "$LOCAL_DIR/vlm_pipeline/"*.py "$SERVER:$REMOTE_DIR/vlm_pipeline/"
scp -i "$KEYFILE" "$LOCAL_DIR/vlm_pipeline/"*.sh "$SERVER:$REMOTE_DIR/vlm_pipeline/" 2>/dev/null || true
scp -i "$KEYFILE" "$LOCAL_DIR/vlm_pipeline/"*.md "$SERVER:$REMOTE_DIR/vlm_pipeline/" 2>/dev/null || true
scp -i "$KEYFILE" "$LOCAL_DIR/vlm_pipeline/"*.txt "$SERVER:$REMOTE_DIR/vlm_pipeline/" 2>/dev/null || true

# Sync pddl folder (needed for planning)
echo ""
echo "Syncing pddl/..."
scp -i "$KEYFILE" "$LOCAL_DIR/pddl/"*.pddl "$SERVER:$REMOTE_DIR/pddl/"

echo ""
echo "=============================================="
echo "   SYNC COMPLETE!"
echo "=============================================="
echo ""
echo "Now SSH into the server and run:"
echo "  ssh $SERVER -i $KEYFILE"
echo "  cd ~/TAMP-PDDL"
echo "  python -m vlm_pipeline.vlm_server --port 8000"
