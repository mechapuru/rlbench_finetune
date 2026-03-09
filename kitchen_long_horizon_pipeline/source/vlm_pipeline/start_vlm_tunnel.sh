#!/bin/bash
# =============================================================================
# start_vlm_tunnel.sh - Start SSH tunnel to VLM server
# =============================================================================
# Usage: ./start_vlm_tunnel.sh
#
# This creates an SSH tunnel from your laptop to the GPU server.
# Keep this terminal open while using the VLM!
# =============================================================================

SERVER="long-horizon@gvlab2.iiit.ac.in"
KEYFILE="$HOME/.ssh/id_ed25519"
LOCAL_PORT=8000
REMOTE_PORT=8000

echo "=============================================="
echo "   VLM SSH TUNNEL"
echo "=============================================="
echo "Server: $SERVER"
echo "Keyfile: $KEYFILE"
echo "Tunnel: localhost:$LOCAL_PORT -> remote:$REMOTE_PORT"
echo "=============================================="
echo ""
echo "Starting SSH tunnel..."
echo "Press Ctrl+C to stop the tunnel."
echo ""

# Check if keyfile exists
if [ ! -f "$KEYFILE" ]; then
    echo "ERROR: Keyfile not found at $KEYFILE"
    echo "Please ensure your SSH key is at ~/keyfile"
    exit 1
fi

# Start the tunnel
ssh -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${SERVER} -i ${KEYFILE} -N -v

# -L = Local port forwarding
# -N = No remote command (just tunnel)
# -v = Verbose (shows connection status)
