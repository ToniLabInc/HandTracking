"""The camera and the model: frames from a V4L2 camera, through MediaPipe's GestureRecognizer, to ``Hand``s.

MediaPipe (Apache 2.0) runs its hand detector, landmark model and gesture classifier on the CPU with XNNPACK, in VIDEO
mode (it tracks hands from frame to frame, re-detecting only when needed). Frames never leave the process, except as
a small picture for the live view while a client asks for one (Orion's HUD, with its camera switched on).

MediaPipe 1.0's package imports matplotlib (for a 3-D plotting helper) and certifi (to download models) at import
time. Neither is used here, so neither is installed: small stand-ins take their place unless the real ones exist.
"""

import importlib.util
import logging
import os
import sys
import time
import types
from typing import Any, Final

from handtracker import config
from handtracker.gestures import Hand

log = logging.getLogger("handtracker.vision")

CA_BUNDLE: Final = "/etc/ssl/certs/ca-certificates.crt"


def _stand_in(name: str, **attrs: Any) -> None:
    """Register an empty module under ``name`` unless the real package is installed."""
    if name in sys.modules:
        return
    top = name.split(".", 1)[0]
    if top not in sys.modules and importlib.util.find_spec(top) is not None:
        return
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def import_mediapipe() -> types.SimpleNamespace:
    """Import the parts of MediaPipe the tracker uses (with the stand-ins for its unused imports).

    Returns:
        A namespace with ``gesture_recognizer``, ``base_options``, ``image`` and ``running_mode``.
    """
    _stand_in("matplotlib")
    _stand_in("matplotlib.pyplot")
    _stand_in("certifi", where=lambda: CA_BUNDLE)
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision import gesture_recognizer
    from mediapipe.tasks.python.vision.core import image, vision_task_running_mode

    return types.SimpleNamespace(gesture_recognizer=gesture_recognizer, base_options=base_options, image=image,
                                 running_mode=vision_task_running_mode.VisionTaskRunningMode)


def hands_from_result(result: Any) -> list[Hand]:
    """The hands in a GestureRecognizerResult, as the engine takes them.

    Args:
        result: MediaPipe's result: per hand, ``gestures`` and ``handedness`` (lists of categories, best first) and
            ``hand_landmarks`` (21 normalised landmarks).

    Returns:
        One ``Hand`` per detected hand, in the result's order.
    """
    hands = []
    for i, landmarks in enumerate(result.hand_landmarks):
        gesture = result.gestures[i][0] if i < len(result.gestures) and result.gestures[i] else None
        handed = result.handedness[i][0] if i < len(result.handedness) and result.handedness[i] else None
        hands.append(Hand(
            landmarks=[(float(p.x), float(p.y), float(p.z or 0.0)) for p in landmarks],
            handedness=str(handed.category_name) if handed is not None else "Right",
            gesture=str(gesture.category_name) if gesture is not None else "None",
            gesture_score=float(gesture.score) if gesture is not None else 0.0,
        ))
    return hands


class Recognizer:
    """MediaPipe's GestureRecognizer, CPU (XNNPACK), set as model.* asks, looking for ``num_hands`` hands.

    Args:
        model: the model settings.
        num_hands: how many hands to look for.
        video: VIDEO mode (frames of a stream, with timestamps); otherwise IMAGE mode (unrelated pictures).
    """

    def __init__(self, model: config.Model, num_hands: int, *, video: bool = True) -> None:
        mp = import_mediapipe()
        self._mp = mp
        self.video = video
        options = mp.gesture_recognizer.GestureRecognizerOptions(
            base_options=mp.base_options.BaseOptions(model_asset_path=str(model.path),
                                                     delegate=mp.base_options.BaseOptions.Delegate.CPU),
            running_mode=mp.running_mode.VIDEO if video else mp.running_mode.IMAGE,
            num_hands=num_hands, min_hand_detection_confidence=model.min_detection,
            min_hand_presence_confidence=model.min_presence, min_tracking_confidence=model.min_tracking)
        self._task = mp.gesture_recognizer.GestureRecognizer.create_from_options(options)
        self._last_ts = -1

    def __call__(self, rgb: Any, ts_ms: int = 0) -> list[Hand]:
        """Recognise the hands in one RGB frame (a contiguous uint8 array, height x width x 3).

        Args:
            rgb: the frame.
            ts_ms: its timestamp in milliseconds (VIDEO mode; made strictly increasing if it isn't).

        Returns:
            The hands found.
        """
        image = self._mp.image.Image(image_format=self._mp.image.ImageFormat.SRGB, data=rgb)
        if not self.video:
            return hands_from_result(self._task.recognize(image))
        ts_ms = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts_ms
        return hands_from_result(self._task.recognize_for_video(image, ts_ms))

    def close(self) -> None:
        """Release the model."""
        self._task.close()


class Scheduler:
    """Picks the recognizer for each frame, to keep the cost of the palm detector down.

    MediaPipe runs its palm detector on every frame while it tracks fewer hands than it looks for, so a two-hand
    recognizer costs a detection plus a landmark pass per frame (on a Pi 5, about 35 + 30 ms) all the time one hand is
    up. So: with no hand in view, the two-hand recognizer looks for hands; once it finds one, a one-hand recognizer
    tracks it (no detection: about 30 ms), and every ``wide_every``-th frame the two-hand recognizer looks again, to
    catch a second hand within a few frames; with two hands, the two-hand recognizer tracks both (no detection either).

    For ``sticky_ms`` after two hands were seen, the two-hand recognizer stays on every frame even while it finds only
    one: two hands closing in or spreading apart often lose one for a few frames, and a one-hand recognizer would keep
    it lost (that cost the two-hand gestures before).

    Args:
        wide: the recognizer for up to two hands.
        narrow: the one-hand recognizer (None: always use ``wide``).
        wide_every: in one-hand mode, every this many frames go to ``wide``.
        sticky_ms: after two hands were seen, ``wide`` takes every frame this long.
    """

    def __init__(self, wide: Any, narrow: Any, *, wide_every: int, sticky_ms: int) -> None:
        self.wide, self.narrow, self.wide_every, self.sticky_ms = wide, narrow, wide_every, sticky_ms
        self.mode = "search"  # search (no hand) | one | two
        self._since_wide = 0
        self._two_until = -1

    def __call__(self, rgb: Any, ts_ms: int) -> list[Hand]:
        """The hands in one frame."""
        sticky = ts_ms < self._two_until
        if self.narrow is not None and self.mode == "one" and not sticky and self._since_wide < self.wide_every - 1:
            self._since_wide += 1
            hands = self.narrow(rgb, ts_ms)
            if not hands:
                self.mode = "search"  # lost it: the next frame goes to the two-hand recognizer
            return hands
        self._since_wide = 0
        hands = self.wide(rgb, ts_ms)
        if len(hands) >= 2:
            self._two_until = ts_ms + self.sticky_ms
        self.mode = ("search", "one", "two")[min(len(hands), 2)] if self.narrow is not None else "two"
        return hands

    def close(self) -> None:
        """Release both recognizers."""
        for recognizer in (self.wide, self.narrow):
            if recognizer is not None:
                recognizer.close()


# V4L2 ioctls (linux/videodev2.h): struct v4l2_queryctrl is 68 bytes, struct v4l2_control 8.
VIDIOC_QUERYCTRL: Final = 0xC0445624
VIDIOC_S_CTRL: Final = 0xC008561C
CTRL_FLAG_NEXT: Final = 0x80000000
CTRL_FLAG_DISABLED: Final = 0x0001


def control_name(raw: bytes) -> str:
    """A V4L2 control's name as v4l2-ctl writes it ("Exposure, Dynamic Framerate" -> "exposure_dynamic_framerate")."""
    text = raw.split(b"\0", 1)[0].decode("ascii", "replace").lower()
    return "_".join("".join(c if c.isalnum() else " " for c in text).split())


def set_controls(path: str, wanted: dict[str, int]) -> dict[str, int]:
    """Set V4L2 controls by name on a camera (names as ``v4l2-ctl --list-ctrls`` shows them).

    Returns:
        The controls that were set. Unknown or refused ones are logged and skipped.
    """
    import fcntl
    import struct

    if not wanted:
        return {}
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as e:
        log.warning("can't open %s to set its controls: %s", path, e)
        return {}
    done: dict[str, int] = {}
    try:
        ids: dict[str, int] = {}
        query = CTRL_FLAG_NEXT
        while True:
            buf = bytearray(struct.pack("=II32s5i2I", query, 0, b"", 0, 0, 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
            except OSError:
                break
            cid, _, name, *_rest, flags, _r1, _r2 = struct.unpack("=II32s5i2I", buf)
            if not flags & CTRL_FLAG_DISABLED:
                ids[control_name(name)] = cid
            query = cid | CTRL_FLAG_NEXT
        for name, value in wanted.items():
            if name not in ids:
                log.warning("camera %s has no control %r (it has: %s)", path, name, ", ".join(sorted(ids)))
                continue
            try:
                fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("=Ii", ids[name], value))
                done[name] = value
            except OSError as e:
                log.warning("camera %s refused %s=%d: %s", path, name, value, e)
    finally:
        os.close(fd)
    return done


class Camera:
    """A V4L2 camera through OpenCV, set as camera.* asks (size, frame rate, pixel format, V4L2 controls, the frames
    the driver queues); ``grab`` waits for a frame, ``retrieve`` decodes it.

    ``camera.device`` may also be a video file: it is played in a loop at its own frame rate, as if it were the camera
    (to try the whole pipeline on a recorded clip).
    """

    def __init__(self, settings: config.Camera) -> None:
        import cv2

        self._cv2, self.settings = cv2, settings
        self._cap: Any = None
        self.actual: dict[str, Any] = {}
        self._file_fps = 0.0  # a video file's own frame rate (0: a camera)
        self._next = 0.0  # a file's next frame time (monotonic)

    def open(self) -> bool:
        """Open the camera and set its format; False when it can't be opened or delivers nothing."""
        cv2, c = self._cv2, self.settings
        self.close()
        if os.path.isfile(c.device):
            return self._open_file()
        source: int | str = int(c.device) if c.device.isdigit() else c.device
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, c.buffer_frames)
        path = f"/dev/video{source}" if isinstance(source, int) else source
        controls = set_controls(path, c.controls)
        if not cap.grab():
            cap.release()
            return False
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.actual = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       "fps": round(cap.get(cv2.CAP_PROP_FPS), 1),
                       "fourcc": "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00"),
                       "controls": controls}
        self._cap = cap
        return True

    def _open_file(self) -> bool:
        """Open a video file to replay."""
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.settings.device)
        if not cap.isOpened() or not cap.grab():
            cap.release()
            return False
        self._file_fps = round(cap.get(cv2.CAP_PROP_FPS)) or self.settings.fps
        self.actual = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": float(self._file_fps),
                       "file": self.settings.device}
        self._cap, self._next = cap, time.monotonic()
        return True

    @property
    def is_open(self) -> bool:
        """Whether the camera is open."""
        return self._cap is not None

    def grab(self) -> bool:
        """Wait for the next frame (not decoded); False when the camera failed."""
        if self._cap is None:
            return False
        if not self._file_fps:
            return bool(self._cap.grab())
        self._next += 1 / self._file_fps  # a file plays at its own pace, looping
        time.sleep(max(0.0, self._next - time.monotonic()))
        if self._cap.grab():
            return True
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
        return bool(self._cap.grab())

    def retrieve(self) -> Any:
        """Decode the last grabbed frame (BGR), or None."""
        if self._cap is None:
            return None
        ok, frame = self._cap.retrieve()
        return frame if ok else None

    def close(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------------------------------- as served
def open_recognizer(cfg: config.TrackerConfig) -> Scheduler:
    """The recognizer as the tracker runs it: two-hand and one-hand recognizers, scheduled (see ``Scheduler``)."""
    m = cfg.model
    return Scheduler(Recognizer(m, m.num_hands), Recognizer(m, 1) if m.num_hands > 1 else None,
                     wide_every=m.wide_every, sticky_ms=round(m.two_hand_sticky_s * 1000))


def prepare(frame: Any, cfg: config.TrackerConfig, cv2: Any) -> Any:
    """A camera frame as the model takes it: scaled to processing.infer_width, and mirrored when camera.mirror."""
    h, w = frame.shape[:2]
    width = cfg.processing.infer_width
    if width and width < w:
        frame = cv2.resize(frame, (width, round(h * width / w)),
                           interpolation=cv2.INTER_AREA if width <= w / 2 else cv2.INTER_LINEAR)
    return cv2.flip(frame, 1) if cfg.camera.mirror else frame
