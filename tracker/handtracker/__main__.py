"""Entry point: ``python -m handtracker COMMAND --config FILE``; FILE is the Pi's settings, laid over defaults.yaml.

- ``serve``: the tracker (camera loop + MCP endpoint).
- ``health``: GET /health of the tracker on this machine (Docker's health check): exit 0 when it is ok.
- ``recognize IMAGE...``: run the recognizer, as served, on photos (IMAGE mode) and print what it sees, one JSON line
  per photo. ``--fit`` first places each photo in a camera-sized frame, as a webcam would see a hand.
- ``bench``: time decoding, scaling and the model at several model input widths, on live camera frames (stop the
  tracker first: one process at a time can use the camera) or on photos (``--images``), with the served recognizer.

For ``recognize`` and ``bench`` the --config file may be left out: defaults.yaml alone is used.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from handtracker import config
from handtracker.config import TrackerConfig

log = logging.getLogger("handtracker")
QUIET: tuple[str, ...] = ("mcp", "uvicorn", "sse_starlette")  # libraries logged at warnings only


def setup_logging(level: str) -> None:
    """Log to stdout, one line per message (Docker keeps them)."""
    logging.basicConfig(level=level, stream=sys.stdout, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for name in QUIET:
        logging.getLogger(name).setLevel(max(logging.WARNING, logging.getLevelName(level)))


def health(cfg: TrackerConfig) -> int:
    """0 when the tracker on this machine says it is ok, 1 otherwise."""
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(cfg.server.host, cfg.server.host)  # every address: this one
    url = f"http://{f'[{host}]' if ':' in host else host}:{cfg.server.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=4) as answer:
            return 0 if answer.status == 200 else 1
    except (urllib.error.URLError, OSError) as e:
        print(f"{url}: {e}", file=sys.stderr)
        return 1


def recognize(cfg: TrackerConfig, paths: list[Path], fit: bool) -> int:
    """Print what the recognizer sees in each photo."""
    import cv2
    import numpy as np

    from handtracker.vision import Recognizer

    recognizer = Recognizer(cfg.model, cfg.model.num_hands, video=False)
    width, height = cfg.camera.width, cfg.camera.height
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(json.dumps({"image": str(path), "error": "can't read it"}))
            continue
        if fit:  # as a webcam would see it: the photo scaled into a camera frame, 3/4 of its height
            scale = min(0.75 * height / bgr.shape[0], 0.95 * width / bgr.shape[1])
            small = cv2.resize(bgr, (max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            frame = np.full((height, width, 3), int(np.median(bgr)), np.uint8)
            y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
            frame[y:y + small.shape[0], x:x + small.shape[1]] = small
            bgr = frame
        start = time.perf_counter()
        hands = recognizer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ms = (time.perf_counter() - start) * 1000
        print(json.dumps({"image": path.name, "size": f"{bgr.shape[1]}x{bgr.shape[0]}", "ms": round(ms, 1),
                          "hands": [{"gesture": h.gesture, "score": round(h.gesture_score, 2),
                                     "handedness": h.handedness} for h in hands]}))
    recognizer.close()
    return 0


def bench(cfg: TrackerConfig, seconds: float, widths: list[int], images: list[Path]) -> int:
    """Time each stage at several model input widths: on live camera frames, or on photos fed as a video."""
    import itertools

    import cv2

    from handtracker.vision import Camera, open_recognizer, prepare

    camera = None
    if images:
        frames = [cv2.imread(str(p)) for p in images]
        frames = [cv2.resize(f, (cfg.camera.width, cfg.camera.height)) for f in frames if f is not None]
        if not frames:
            print("none of the images could be read", file=sys.stderr)
            return 1
        source = itertools.cycle(frames)
    else:
        camera = Camera(cfg.camera)
        if not camera.open():
            print(f"can't open {cfg.camera.device}", file=sys.stderr)
            return 1
        print(json.dumps({"camera": camera.actual}))
    for width in widths:
        served = cfg.model_copy(update={"processing": cfg.processing.model_copy(update={"infer_width": width})})
        recognizer = open_recognizer(served)
        decode = scale = model = 0.0
        hands_seen = n = 0
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            t0 = time.perf_counter()
            if camera is not None:
                if not camera.grab():
                    break
                t0 = time.perf_counter()
                frame = camera.retrieve()
            else:
                frame = next(source)
            t1 = time.perf_counter()
            rgb = cv2.cvtColor(prepare(frame, served, cv2), cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter()
            hands = recognizer(rgb, int((time.monotonic() - start) * 1000))
            t3 = time.perf_counter()
            decode, scale, model, n = decode + t1 - t0, scale + t2 - t1, model + t3 - t2, n + 1
            hands_seen += len(hands)
        elapsed = time.monotonic() - start
        recognizer.close()
        if n:
            print(json.dumps({"width": width or cfg.camera.width, "num_hands": cfg.model.num_hands, "frames": n,
                              "fps": round(n / elapsed, 1), "decode_ms": round(1000 * decode / n, 2),
                              "scale_flip_rgb_ms": round(1000 * scale / n, 2), "model_ms": round(1000 * model / n, 2),
                              "hands_per_frame": round(hands_seen / n, 2)}))
    if camera is not None:
        camera.close()
    return 0


def main() -> None:
    """Parse the command line and run."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="the Pi's settings file, laid over defaults.yaml")
    parser = argparse.ArgumentParser(prog="handtracker", description="ORION hand-gesture tracker")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", parents=[common], help="run the tracker")
    sub.add_parser("health", parents=[common], help="exit 0 when the tracker here is ok")
    rec = sub.add_parser("recognize", parents=[common], help="recognise gestures in photos")
    rec.add_argument("images", nargs="+", type=Path)
    rec.add_argument("--fit", action="store_true", help="place each photo in a camera-sized frame first")
    ben = sub.add_parser("bench", parents=[common], help="time the stages on camera frames or photos")
    ben.add_argument("--seconds", type=float, default=8.0)
    ben.add_argument("--widths", nargs="+", type=int, default=[640, 480, 320],
                     help="model input widths (0: as captured)")
    ben.add_argument("--images", nargs="*", type=Path, default=[], help="photos to loop through instead of the camera")
    args = parser.parse_args()
    if args.command in ("serve", "health") and args.config is None:
        parser.error(f"{args.command} needs --config (the Pi's settings file)")
    try:
        cfg = config.load(args.config)
    except (OSError, config.ConfigError) as e:
        print(f"handtracker: {e}", file=sys.stderr)
        sys.exit(2)
    if args.command == "health":
        sys.exit(health(cfg))
    setup_logging(cfg.logging.level)
    if args.command == "recognize":
        sys.exit(recognize(cfg, args.images, args.fit))
    if args.command == "bench":
        sys.exit(bench(cfg, args.seconds, args.widths, args.images))
    from handtracker.server import serve

    try:
        sys.exit(asyncio.run(serve(cfg, args.config)))
    except config.ConfigError as e:
        log.error("%s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
