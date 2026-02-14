"""
WebSocket Telemetry Server for HEMOC-SGF Physics Engine

Wraps the OpticalKirigamiEncoder in a WebSocket service that streams
real-time physics telemetry as JSON payloads matching the
HemocPhysicsPayload schema consumed by HemocPythonBridge.ts on the
frontend.

Usage:
    python websocket_server.py [--host 0.0.0.0] [--port 8765] [--fps 30]

Protocol:
    Server → Client (per frame):
    {
        "frame_id": int,
        "timestamp": float,
        "moire": { "period", "contrast", "dominant_frequency", ... },
        "kirigami": { "petal_rotations", "lattice_stress", "cell_distribution", ... },
        "reservoir": { "entropy", "lyapunov", "memory_capacity", "kernel_weights" },
        "talbot": { "gap_mode", "logic_polarity", "gap_distance" },
        "actuators": [{ "tip", "tilt", "piston" }, ...],
        "feature_vector": [float, ...]
    }

    Client → Server (commands):
    { "command": "set_mode", "angle": float, "gap": float }
    { "command": "pause" | "resume" | "reset" }
"""

import asyncio
import json
import time
import math
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing the real engine; fall back to a synthetic telemetry source
# ---------------------------------------------------------------------------
try:
    from physics.moire_interference import MoireInterference
    from physics.talbot_resonator import TalbotResonator
    from physics.trilatic_lattice import TrilaticLattice
    from kirigami.kirigami_sheet import KirigamiSheet
    from kirigami.tristable_cell import TristableCell
    from control.tripole_actuator import TripoleActuator
    from reservoir.criticality import edge_of_chaos_metric
    from telemetry.metrics import extract_telemetry
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    logger.warning("HEMOC-SGF engine modules not found; using synthetic telemetry")


@dataclass
class TelemetryFrame:
    """Matches the HemocPhysicsPayload TypeScript interface."""
    frame_id: int
    timestamp: float
    moire: dict
    kirigami: dict
    reservoir: dict
    talbot: dict
    actuators: list
    feature_vector: Optional[list] = None


class SyntheticEngine:
    """
    Generates plausible physics telemetry without the real engine.
    Useful for frontend development and integration testing.
    """

    def __init__(self):
        self.frame_count = 0
        self.angle = 0.0
        self.gap_distance = 25.0
        self.paused = False
        self.petal_angles = [0.0, 0.0, 0.0]

    def step(self) -> TelemetryFrame:
        self.frame_count += 1
        t = self.frame_count / 30.0  # Assume 30 FPS

        # Evolve petal rotations (three petals with different rates)
        for i in range(3):
            rate = [0.3, 0.5, 0.7][i]
            self.petal_angles[i] += rate * 0.033
            self.petal_angles[i] %= (2 * math.pi)

        # Moiré contrast oscillates with interference
        contrast = 0.5 + 0.4 * math.sin(t * 0.8) * math.cos(t * 1.3)
        period = 10.0 + 5.0 * math.sin(t * 0.3)
        freq = 1000.0 / period if period > 0 else 0

        # Lattice stress from petal divergence
        stress_vals = [
            math.sin(self.petal_angles[0]) * 0.5,
            math.cos(self.petal_angles[1]) * 0.3,
            math.sin(self.petal_angles[2]) * 0.4,
            0, 0, 0, 0, 0, 0
        ]

        # Cell distribution shifts with angle
        flat = 0.3 + 0.2 * math.sin(t * 0.5)
        half = 0.4 + 0.1 * math.cos(t * 0.7)
        full = max(0, 1.0 - flat - half)

        # Reservoir entropy (peaks near edge-of-chaos)
        entropy = 4.0 + 2.0 * math.sin(t * 0.2) + 0.5 * math.sin(t * 1.1)
        lyapunov = 0.01 + 0.05 * math.sin(t * 0.15)
        memory_cap = 0.6 + 0.3 * math.cos(t * 0.1)

        # Talbot mode alternates
        is_integer = int(t * 0.1) % 2 == 0

        return TelemetryFrame(
            frame_id=self.frame_count,
            timestamp=time.time() * 1000,
            moire={
                "period": period,
                "contrast": max(0, min(1, contrast)),
                "dominant_frequency": freq,
                "mean_intensity": 0.5 + 0.1 * math.sin(t),
                "max_intensity": 0.9,
                "min_intensity": 0.1,
            },
            kirigami={
                "petal_rotations": list(self.petal_angles),
                "lattice_stress": stress_vals,
                "cell_distribution": [
                    max(0, min(1, flat)),
                    max(0, min(1, half)),
                    max(0, min(1, full)),
                ],
                "operating_angle": self.angle,
            },
            reservoir={
                "entropy": max(0, entropy),
                "lyapunov": lyapunov,
                "memory_capacity": max(0, min(1, memory_cap)),
                "kernel_weights": [math.exp(-0.1 * k) for k in range(8)],
            },
            talbot={
                "gap_mode": "integer" if is_integer else "half_integer",
                "logic_polarity": "positive" if is_integer else "negative",
                "gap_distance": self.gap_distance,
            },
            actuators=[
                {"tip": 0.0, "tilt": 0.0, "piston": 0.0},
                {"tip": 0.0, "tilt": 0.0, "piston": 0.0},
            ],
            feature_vector=[contrast, freq / 100, entropy / 8, lyapunov],
        )

    def handle_command(self, cmd: dict) -> None:
        action = cmd.get("command", "")
        if action == "set_mode":
            self.angle = cmd.get("angle", self.angle)
            self.gap_distance = cmd.get("gap", self.gap_distance)
        elif action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "reset":
            self.__init__()


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

class HemocTelemetryServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, fps: int = 30):
        self.host = host
        self.port = port
        self.interval = 1.0 / fps
        self.engine = SyntheticEngine()
        self.clients: set = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        logger.info(f"Client connected ({len(self.clients)} total)")
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    self.engine.handle_command(cmd)
                except json.JSONDecodeError:
                    pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected ({len(self.clients)} total)")

    async def broadcast_loop(self):
        while True:
            if not self.engine.paused and self.clients:
                frame = self.engine.step()
                payload = json.dumps(asdict(frame))
                disconnected = set()
                for ws in self.clients:
                    try:
                        await ws.send(payload)
                    except Exception:
                        disconnected.add(ws)
                self.clients -= disconnected
            await asyncio.sleep(self.interval)

    async def run(self):
        try:
            import websockets
        except ImportError:
            logger.error("Install websockets: pip install websockets")
            return

        logger.info(f"HEMOC Telemetry Server starting on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await self.broadcast_loop()


def main():
    parser = argparse.ArgumentParser(description="HEMOC-SGF WebSocket Telemetry Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = HemocTelemetryServer(host=args.host, port=args.port, fps=args.fps)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
