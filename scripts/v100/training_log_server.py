#!/usr/bin/env python3
"""
Training Log Server - WebSocket server that tails training logs and pushes parsed metrics.

Watches a training log file, parses step/reward/tool_call_success_rate lines,
and streams both raw log lines and structured metrics to the frontend.

Usage:
    python training_log_server.py --log-file /path/to/training.log [--port 8766] [--interval 1]

The server pushes two types of messages:
  1. {"type": "log", "line": "...", "timestamp": "..."}
  2. {"type": "metric", "step": N, "reward": 0.xx, "tool_success": 0.xx, ...}
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


# Patterns to extract metrics from training logs
# Adapt these to match your actual log format
METRIC_PATTERNS = {
    # Pattern: step=100, reward_mean=0.325, ...
    "step": re.compile(r"(?:step|global_step)[=:\s]+(\d+)"),
    "reward": re.compile(r"(?:reward(?:/mean|_mean)?)[=:\s]+([\d.]+)"),
    "tool_success": re.compile(r"(?:tool_call_success_rate|tool_success)[=:\s]+([\d.]+)"),
    "kl": re.compile(r"(?:kl_divergence|kl)[=:\s]+([\d.]+)"),
    "grad_norm": re.compile(r"(?:grad_norm)[=:\s]+([\d.]+)"),
    "loss": re.compile(r"(?:policy_loss|loss)[=:\s]+([\d.eE\-]+)"),
}


def parse_metric_line(line: str) -> dict | None:
    """Try to extract metrics from a log line. Returns dict if any metric found."""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(line)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass

    # Only return if we got at least a step number
    if "step" in metrics:
        metrics["step"] = int(metrics["step"])
        return metrics
    return None


class LogTailer:
    """Tails a log file, yielding new lines as they appear."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.position = 0
        self._check_file()

    def _check_file(self):
        """Initialize position to end of file (only stream new content)."""
        if os.path.exists(self.filepath):
            self.position = os.path.getsize(self.filepath)

    def get_new_lines(self) -> list[str]:
        """Read any new lines since last check."""
        lines = []
        try:
            if not os.path.exists(self.filepath):
                return lines

            current_size = os.path.getsize(self.filepath)

            # File was truncated/rotated
            if current_size < self.position:
                self.position = 0

            if current_size == self.position:
                return lines

            with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.position)
                new_content = f.read()
                self.position = f.tell()

            lines = new_content.splitlines()
        except (IOError, OSError):
            pass
        return lines


# Connected clients
clients: set = set()
# Recent metrics buffer (for new clients to catch up)
metrics_buffer: list = []
MAX_BUFFER = 500


async def register(websocket):
    """Register a new client and send buffered metrics."""
    clients.add(websocket)
    print(f"[Log Server] Client connected: {websocket.remote_address} (total: {len(clients)})")
    # Send buffered metrics to new client
    if metrics_buffer:
        await websocket.send(json.dumps({
            "type": "history",
            "metrics": metrics_buffer,
        }))


async def unregister(websocket):
    clients.discard(websocket)
    print(f"[Log Server] Client disconnected (total: {len(clients)})")


async def broadcast(message: dict):
    """Send message to all connected clients."""
    if not clients:
        return
    data = json.dumps(message)
    disconnected = set()
    for ws in clients:
        try:
            await ws.send(data)
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(ws)
    for ws in disconnected:
        clients.discard(ws)


async def tail_log(filepath: str, interval: float = 1.0):
    """Main loop: tail the log file and broadcast new lines/metrics."""
    tailer = LogTailer(filepath)
    print(f"[Log Server] Tailing: {filepath}")
    print(f"[Log Server] Poll interval: {interval}s")

    while True:
        lines = tailer.get_new_lines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Broadcast raw log line
            await broadcast({
                "type": "log",
                "line": line,
                "timestamp": datetime.now().isoformat(),
            })

            # Try to parse metrics
            metrics = parse_metric_line(line)
            if metrics:
                metrics["timestamp"] = datetime.now().isoformat()
                metrics_buffer.append(metrics)
                if len(metrics_buffer) > MAX_BUFFER:
                    metrics_buffer.pop(0)

                await broadcast({
                    "type": "metric",
                    **metrics,
                })

        await asyncio.sleep(interval)


async def handler(websocket):
    """WebSocket connection handler."""
    await register(websocket)
    try:
        # Keep connection alive, handle client messages if needed
        async for message in websocket:
            # Client can send commands like {"action": "get_history"}
            try:
                data = json.loads(message)
                if data.get("action") == "get_history":
                    await websocket.send(json.dumps({
                        "type": "history",
                        "metrics": metrics_buffer,
                    }))
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await unregister(websocket)


async def main(log_file: str, port: int = 8766, interval: float = 1.0):
    """Start the WebSocket server and log tailer."""
    if not HAS_WEBSOCKETS:
        print("ERROR: 'websockets' package not installed.")
        print("Install with: pip install websockets")
        print("")
        print("Falling back to console mode (printing parsed metrics)...")
        print("=" * 60)
        tailer = LogTailer(log_file)
        while True:
            lines = tailer.get_new_lines()
            for line in lines:
                metrics = parse_metric_line(line.strip())
                if metrics:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {metrics}")
            time.sleep(interval)
        return

    print(f"[Log Server] Starting on ws://0.0.0.0:{port}")
    print(f"[Log Server] Watching: {log_file}")

    # Start log tailer as background task
    asyncio.create_task(tail_log(log_file, interval))

    async with websockets.serve(handler, "0.0.0.0", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Log WebSocket Server")
    parser.add_argument("--log-file", type=str, required=True, help="Path to training log file to tail")
    parser.add_argument("--port", type=int, default=8766, help="WebSocket port (default: 8766)")
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"[Log Server] Warning: {args.log_file} does not exist yet. Will watch for creation.")

    try:
        asyncio.run(main(log_file=args.log_file, port=args.port, interval=args.interval))
    except KeyboardInterrupt:
        print("\n[Log Server] Stopped.")
