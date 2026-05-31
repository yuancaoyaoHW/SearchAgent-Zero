#!/usr/bin/env python3
"""
GPU Monitor Server - Lightweight WebSocket server for real-time GPU metrics.

Pushes nvidia-smi data to the training-tracker frontend every 5 seconds.

Usage:
    python gpu_monitor_server.py [--port 8765] [--interval 5]

Then in the frontend, connect to ws://localhost:8765 for live GPU data.
"""

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


def get_gpu_metrics():
    """Query nvidia-smi for GPU metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,power.limit,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 10:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temperature": float(parts[2]) if parts[2] != "[N/A]" else None,
                    "gpu_util": float(parts[3]) if parts[3] != "[N/A]" else None,
                    "mem_util": float(parts[4]) if parts[4] != "[N/A]" else None,
                    "mem_used_mb": float(parts[5]) if parts[5] != "[N/A]" else None,
                    "mem_total_mb": float(parts[6]) if parts[6] != "[N/A]" else None,
                    "power_w": float(parts[7]) if parts[7] != "[N/A]" else None,
                    "power_limit_w": float(parts[8]) if parts[8] != "[N/A]" else None,
                    "clock_mhz": float(parts[9]) if parts[9] != "[N/A]" else None,
                })
        return {
            "timestamp": datetime.now().isoformat(),
            "gpus": gpus,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


async def gpu_monitor_handler(websocket, interval=5):
    """WebSocket handler that pushes GPU metrics at regular intervals."""
    print(f"[GPU Monitor] Client connected: {websocket.remote_address}")
    try:
        while True:
            metrics = get_gpu_metrics()
            if metrics:
                await websocket.send(json.dumps(metrics))
            await asyncio.sleep(interval)
    except websockets.exceptions.ConnectionClosed:
        print(f"[GPU Monitor] Client disconnected: {websocket.remote_address}")


async def main(port=8765, interval=5):
    """Start the WebSocket server."""
    if not HAS_WEBSOCKETS:
        print("ERROR: 'websockets' package not installed.")
        print("Install with: pip install websockets")
        print("")
        print("Falling back to console mode (printing GPU metrics every 5s)...")
        print("=" * 60)
        while True:
            metrics = get_gpu_metrics()
            if metrics and "gpus" in metrics:
                print(f"\n[{metrics['timestamp']}]")
                for gpu in metrics["gpus"]:
                    print(
                        f"  GPU {gpu['index']}: "
                        f"{gpu['gpu_util']:5.1f}% util | "
                        f"{gpu['mem_used_mb']:.0f}/{gpu['mem_total_mb']:.0f} MB | "
                        f"{gpu['temperature']}°C | "
                        f"{gpu['power_w']:.0f}W"
                    )
            time.sleep(interval)
        return

    print(f"[GPU Monitor] Starting WebSocket server on ws://0.0.0.0:{port}")
    print(f"[GPU Monitor] Pushing metrics every {interval}s")
    print(f"[GPU Monitor] Connect from frontend or test with:")
    print(f"  python -c \"import asyncio, websockets; asyncio.run(websockets.connect('ws://localhost:{port}'))\"")

    async with websockets.serve(
        lambda ws: gpu_monitor_handler(ws, interval),
        "0.0.0.0",
        port,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Monitor WebSocket Server")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds (default: 5)")
    args = parser.parse_args()

    try:
        asyncio.run(main(port=args.port, interval=args.interval))
    except KeyboardInterrupt:
        print("\n[GPU Monitor] Stopped.")
