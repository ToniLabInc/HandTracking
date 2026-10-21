"""The camera loop: grab frames, run the chosen ones through the model and the engine, hand the events on; and the
switch that runs it only while it is wanted.

The loop runs on its own thread (OpenCV and MediaPipe block). Frames are grabbed at the camera's rate, but only decoded
and processed at ``processing.fps`` while a hand is in view, at ``idle_fps`` once none has been for ``idle_after_s``,
and at ``deep_idle_fps`` once none has been for ``deep_idle_after_s``: frames in between are dropped undecoded. While
the live view is watched, every frame runs at the full rate (``full_rate_while_watched``), so the view stays smooth.
Events go to ``publish`` (the MCP server, via the event loop).

While a client watches the live view (``View``), up to ``live_view.fps`` processed frames a second also become a
frame: each hand's landmarks, its shape and how long it has held it; and, while a client asks for the picture, a small
JPEG of what the camera sees (``live_view.picture_fps`` a second). Nobody watching, none of it is made.

The camera is on only on demand (``CameraSwitch``): while a client follows the tracker (an open subscription stream)
and gestures aren't paused. When the last client leaves it stays on for ``camera.idle_grace_s`` (a reconnect doesn't
blink it), then the loop ends: the camera is released (its light goes off) and the model's memory freed. A pause turns
it off at once. Off, the tracker only answers the MCP endpoint: no frames, next to no CPU.
"""

import base64
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from handtracker.config import TrackerConfig
from handtracker.gestures import Engine, Event, Frame, Hand, HandView
from handtracker.vision import Camera, open_recognizer, prepare

log = logging.getLogger("handtracker.camera")

PAGE: Final = os.sysconf("SC_PAGE_SIZE")
RATE_WINDOW_S: Final = 2.0  # frame rates are measured over this long
TIMES_KEPT: Final = 64  # ...from at most this many frame times
CREDIT_MAX: Final = 1.5  # frames owed at most when processing falls behind (a burst of at most one more)


def rss_mb() -> float:
    """This process's resident memory, MB."""
    return round(int(Path("/proc/self/statm").read_text().split()[1]) * PAGE / 1_048_576, 1)


@dataclass
class Stats:
    """Live measurements, for /health and hands_status (written by the camera thread)."""

    camera: bool = False
    camera_format: dict[str, Any] = field(default_factory=dict)
    frames: int = 0  # processed since start
    fps: float = 0.0  # processed frames per second
    capture_fps: float = 0.0  # frames the camera delivers per second
    inference_ms: float = 0.0  # model time per processed frame (moving average)
    pipeline_ms: float = 0.0  # frame grabbed to events ready (decode, scale, model, engine)
    last_frame: float = 0.0  # monotonic time of the last grabbed frame
    _processed: deque[float] = field(default_factory=lambda: deque(maxlen=TIMES_KEPT))
    _grabbed: deque[float] = field(default_factory=lambda: deque(maxlen=TIMES_KEPT))
    _cpu: tuple[float, float] = (0.0, 0.0)  # (wall, cpu seconds) at the last cpu_percent call

    @staticmethod
    def _rate(times: deque[float], now: float) -> float:
        recent = [t for t in times if now - t <= RATE_WINDOW_S]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0]) if recent[-1] > recent[0] else 0.0

    def grabbed(self, t: float) -> None:
        """A frame arrived."""
        self.last_frame = t
        self._grabbed.append(t)

    def processed(self, t: float, inference_ms: float, pipeline_ms: float) -> None:
        """A frame went through the model and the engine."""
        self.frames += 1
        self._processed.append(t)
        k = 0.1 if self.frames > 10 else 1 / self.frames
        self.inference_ms += k * (inference_ms - self.inference_ms)
        self.pipeline_ms += k * (pipeline_ms - self.pipeline_ms)

    def rates(self, now: float) -> tuple[float, float]:
        """(processed fps, camera fps) over the last two seconds."""
        self.fps = round(self._rate(self._processed, now), 1)
        self.capture_fps = round(self._rate(self._grabbed, now), 1)
        return self.fps, self.capture_fps

    def cpu_percent(self) -> float:
        """This process's CPU use since the last call, in % of one core (all threads)."""
        wall, cpu = time.monotonic(), time.process_time()
        last_wall, last_cpu = self._cpu
        self._cpu = (wall, cpu)
        return round(100 * (cpu - last_cpu) / (wall - last_wall), 1) if last_wall and wall > last_wall else 0.0


class Pace:
    """At most ``rate`` a second on average, a burst of at most ``BURST``: processed frames come at the camera's
    uneven pace (two of three at 20 of 30 fps: 33 and 66 ms apart), and a strict minimum gap would drop half of them.

    Args:
        rate: how many a second.
    """

    BURST: Final = 2.0

    def __init__(self, rate: float) -> None:
        self.rate, self.credit, self.last = rate, 1.0, -math.inf

    def take(self, t: float) -> bool:
        """Whether one may go now (it is counted when it may)."""
        if self.last > -math.inf:
            self.credit = min(self.credit + (t - self.last) * self.rate, self.BURST)
        self.last = t
        if self.credit < 1.0 - 1e-6:
            return False
        self.credit -= 1.0
        return True


class View:
    """Who watches the live view, and its frames: the server sets how many clients want the landmarks and how many
    the camera's picture too; the camera thread reads the counts.

    Args:
        fps: frames a second at most (live_view.fps).
        picture_fps: of which this many carry the picture (live_view.picture_fps).
    """

    def __init__(self, fps: float, picture_fps: float) -> None:
        self.landmarks = 0
        self.images = 0
        self.seq = 0
        self._frames, self._pictures = Pace(fps), Pace(picture_fps)

    @property
    def watched(self) -> bool:
        """Whether anyone watches the view."""
        return self.landmarks > 0 or self.images > 0

    def wanted(self, t: float) -> bool:
        """Whether to make a live-view frame now (a frame then follows: it is counted)."""
        return self.watched and self._frames.take(t)

    def picture_wanted(self, t: float) -> bool:
        """Whether that frame should carry the camera's picture (counted the same way)."""
        return self.images > 0 and self._pictures.take(t)

    def frame(self, *, wall: float, size: tuple[int, int], hands: Sequence[Hand], shown: Sequence[HandView], fps: float,
              mode: str, apart: float | None = None, jpeg: bytes | None = None) -> Event:
        """A live-view frame: each hand's landmarks (x, y in the mirrored image, 0 to 1) with what the engine made of
        it, how far apart two hands are (palm lengths, as hands_apart and hands_together measure it), and the picture
        when given."""
        self.seq += 1
        event: Event = {"seq": self.seq, "ts": round(wall, 3), "w": size[0], "h": size[1], "fps": fps, "mode": mode,
                        "hands": [{"track": v.track, "hand": v.hand, "shape": v.shape, "score": v.score,
                                   "held": v.held, "pts": [[round(x, 3), round(y, 3)] for x, y, _ in h.landmarks]}
                                  for h, v in zip(hands, shown, strict=False)],
                        "apart": None if apart is None else round(apart, 2)}
        if jpeg is not None:
            event["jpg"] = base64.b64encode(jpeg).decode("ascii")
        return event


class Pipeline(threading.Thread):
    """The camera thread.

    Args:
        config: the settings (camera, processing, model and live_view are read).
        engine: the gesture engine.
        publish: called (on this thread) with each frame's events, when there are any.
        stats: where to write the measurements.
        view: who watches the live view.
        publish_frame: called (on this thread) with each live-view frame.
    """

    def __init__(self, config: TrackerConfig, engine: Engine, publish: Callable[[list[Event]], None], stats: Stats, *,
                 view: View, publish_frame: Callable[[Event], None]) -> None:
        super().__init__(name="camera", daemon=True)
        self.config, self.engine, self.publish, self.stats = config, engine, publish, stats
        self.view, self.publish_frame = view, publish_frame
        self.stopping = threading.Event()
        self.failed: str = ""  # set when the camera stayed unusable: the service exits so Docker restarts it
        self.scheduler: Any = None

    def stop(self) -> None:
        """Ask the loop to end."""
        self.stopping.set()

    def run(self) -> None:
        """Open the model and the camera, then process frames until stopped."""
        try:
            self._run()
        except Exception as e:  # the service must notice and exit rather than keep a dead camera
            log.exception("the camera loop failed")
            self.failed = f"{type(e).__name__}: {e}"

    def _run(self) -> None:
        import cv2

        c, p = self.config.camera, self.config.processing
        recognizer = self.scheduler = open_recognizer(self.config)
        camera = Camera(c)
        credit = 0.0
        started = since_ok = time.monotonic()
        try:
            while not self.stopping.is_set():
                if not camera.is_open:
                    if camera.open():
                        log.info("camera %s open: %s", c.device, camera.actual)
                        self.stats.camera, self.stats.camera_format = True, camera.actual
                        credit = 1.0
                    else:
                        self.stats.camera = False
                        if time.monotonic() - since_ok > c.exit_after_s:
                            self.failed = f"no frames from {c.device} for {c.exit_after_s:.0f} s"
                            return
                        log.warning("camera %s unavailable; retrying in %g s", c.device, c.reopen_every_s)
                        self.stopping.wait(c.reopen_every_s)
                        continue
                if not camera.grab():
                    log.warning("camera %s stopped delivering frames; reopening", c.device)
                    camera.close()
                    self.stats.camera = False
                    continue
                t = time.monotonic()
                since_ok = t
                self.stats.grabbed(t)
                _, capture_fps = self.stats.rates(t)
                target = p.fps
                full = p.full_rate_while_watched and self.view.watched
                if p.idle_fps and not full and not self.engine.hand_seen_within(t, p.idle_after_s):
                    target = min(p.idle_fps, p.fps)
                    deep = (t - started > p.deep_idle_after_s
                            and not self.engine.hand_seen_within(t, p.deep_idle_after_s))
                    if p.deep_idle_fps and deep:
                        target = min(p.deep_idle_fps, target)
                credit = min(credit + target / max(capture_fps or c.fps, target), CREDIT_MAX)
                if credit < 1.0:
                    continue  # dropped without decoding
                credit -= 1.0
                frame = camera.retrieve()
                if frame is None:
                    continue
                self._process(frame, t, recognizer, cv2)
        finally:
            camera.close()
            recognizer.close()
            self.stats.camera = False

    def _process(self, frame: Any, t: float, recognizer: Any, cv2: Any) -> None:
        """Scale, mirror, recognise, run the engine, publish; and make a live-view frame if someone watches."""
        frame = prepare(frame, self.config, cv2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        hands = recognizer(rgb, int(t * 1000))
        inference_ms = (time.perf_counter() - start) * 1000
        events = self.engine.update(Frame(t, hands, frame.shape[1] / frame.shape[0]))
        if events:
            self.publish(events)
        self.stats.processed(t, inference_ms, (time.monotonic() - t) * 1000)
        view = self.view
        if view.wanted(t):
            jpeg = None
            if view.picture_wanted(t):
                lv = self.config.live_view
                small = cv2.resize(frame, (lv.picture_width, round(frame.shape[0] * lv.picture_width / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, lv.picture_quality])
                jpeg = buf.tobytes() if ok else None
            self.publish_frame(view.frame(wall=time.time(), size=(frame.shape[1], frame.shape[0]), hands=hands,
                                          shown=self.engine.view, fps=self.stats.fps, mode=recognizer.mode,
                                          apart=self.engine.apart, jpeg=jpeg))


class CameraSwitch:
    """Runs the camera loop while it is wanted and ends it when it isn't: at once on a pause, after ``grace`` seconds
    when the last client left. A new loop waits for the old one to let go of the camera first.

    Args:
        make: makes a fresh camera loop (a thread, not started).
        grace: how long the camera stays on once it is no longer wanted (unless ended at once).
    """

    def __init__(self, make: Callable[[], Any], grace: float) -> None:
        self.make, self.grace = make, grace
        self.pipeline: Any = None  # the running loop
        self.closing: Any = None  # the last loop, still letting go of the camera
        self.since = -math.inf  # monotonic time it last turned on
        self.failed = ""  # why the camera stayed unusable (the service then exits, so Docker restarts it)
        self._unwanted_since: float | None = None

    @property
    def on(self) -> bool:
        """Whether a camera loop runs."""
        return self.pipeline is not None and self.pipeline.is_alive()

    def tick(self, wanted: bool, now: float, *, at_once: bool = False) -> str | None:
        """Bring the camera in line with ``wanted`` (the service calls this often, and on every change).

        Args:
            wanted: whether anyone needs the camera now.
            now: the monotonic time.
            at_once: end the loop now rather than after the grace (a pause).

        Returns:
            "on" or "off" when it switched, else None.
        """
        if self.pipeline is not None and not self.pipeline.is_alive():
            self.failed = self.pipeline.failed or self.failed
            self.pipeline = None
        if self.closing is not None and not self.closing.is_alive():
            self.closing = None
        if wanted:
            self._unwanted_since = None
            if self.pipeline is None and self.closing is None and not self.failed:
                self.pipeline = self.make()
                self.pipeline.start()
                self.since = now
                return "on"
            return None
        if self.pipeline is None:
            self._unwanted_since = None
            return None
        if self._unwanted_since is None:
            self._unwanted_since = now
        if not at_once and now - self._unwanted_since < self.grace:
            return None
        self.pipeline.stop()  # the loop releases the camera and the model as it ends
        self.closing, self.pipeline, self._unwanted_since = self.pipeline, None, None
        return "off"
