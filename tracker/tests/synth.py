"""Synthetic hands for the tests: MediaPipe-like landmarks for a hand at a given place, size and shape."""

from collections.abc import Callable, Sequence

from handtracker.gestures import Engine, Event, Frame, Hand

ASPECT = 4 / 3
# An upright hand, palm to the camera, in palm lengths around (0, 0): wrist to middle knuckle is 1.0. Every finger
# straight.
BASE: Sequence[tuple[float, float]] = (
    (0.0, 0.6),  # 0 wrist
    (-0.35, 0.45), (-0.6, 0.2), (-0.8, 0.0), (-0.95, -0.2),  # thumb
    (-0.3, -0.35), (-0.35, -0.8), (-0.38, -1.1), (-0.4, -1.35),  # index
    (0.0, -0.4), (0.0, -0.9), (0.0, -1.2), (0.0, -1.5),  # middle
    (0.15, -0.37), (0.18, -0.8), (0.2, -1.1), (0.21, -1.35),  # ring
    (0.36, -0.28), (0.42, -0.6), (0.45, -0.85), (0.47, -1.05),  # pinky
)
# Each finger folded: its points from the second on (the knuckle stays), as seen from the front.
FOLDED: Sequence[Sequence[tuple[float, float]]] = (
    ((-0.45, 0.25), (-0.4, 0.0), (-0.25, -0.3)),  # thumb across the palm, its tip on the index knuckle
    ((-0.33, -0.6), (-0.3, -0.45), (-0.27, -0.28)),
    ((0.0, -0.65), (0.02, -0.5), (0.03, -0.32)),
    ((0.17, -0.6), (0.18, -0.45), (0.18, -0.3)),
    ((0.4, -0.48), (0.38, -0.35), (0.36, -0.22)),
)
FIRST: Sequence[int] = (2, 6, 10, 14, 18)  # the first folding point of each finger
# The OK sign: the index bent into a ring with the thumb, the other fingers up.
RING: Sequence[tuple[int, tuple[float, float]]] = (
    (2, (-0.6, 0.2)), (3, (-0.7, -0.2)), (4, (-0.62, -0.58)),
    (6, (-0.4, -0.7)), (7, (-0.55, -0.75)), (8, (-0.6, -0.6)),
)


def hand(x: float, y: float, *, size: float = 0.15, gesture: str = "None", score: float = 0.9,
         handedness: str = "Right", fingers: str = "11111", pinch: bool = False, ok: bool = False) -> Hand:
    """A hand whose wrist-to-knuckle line is centred at (x, y) in image-height units (mirrored view).

    Args:
        x: horizontal position, image-height units (0 .. 4/3 across a 4:3 image).
        y: vertical position, image-height units (0 at the top).
        size: palm length, image-height units.
        gesture: the canned label the classifier reports ("Open_Palm", "None", ...).
        score: its confidence.
        handedness: "Left" or "Right".
        fingers: which fingers are straight (1) or folded (0): thumb, index, middle, ring, pinky.
        pinch: put the thumb tip against the index tip.
        ok: the OK sign (the thumb and the bent index making a ring; the other fingers as ``fingers`` says).
    """
    pts = list(BASE)
    for finger, state in enumerate(fingers):
        if state == "0":
            for k, point in enumerate(FOLDED[finger]):
                pts[FIRST[finger] + k] = point
    if ok:
        for i, point in RING:
            pts[i] = point
    if pinch:
        ix, iy = pts[8]
        pts[4] = (ix + 0.05, iy)
    scaled = [(x + u * size, y + v * size) for u, v in pts]
    return Hand(landmarks=[(px / ASPECT, py, 0.0) for px, py in scaled], handedness=handedness,
                gesture=gesture, gesture_score=score)


class Clock:
    """A fake wall clock that follows the simulation."""

    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


class Sim:
    """Feeds an engine frames at a fixed rate and collects its events."""

    def __init__(self, engine: Engine, clock: Clock, *, fps: float = 20, t0: float = 1000.0) -> None:
        self.engine, self.clock, self.fps, self.t = engine, clock, fps, t0
        self.events: list[Event] = []

    def run(self, seconds: float, hands: Callable[[float], Sequence[Hand]] | Sequence[Hand] = ()) -> list[Event]:
        """Run ``seconds`` of frames; ``hands`` is a fixed list or a function of the time since this run started."""
        out: list[Event] = []
        start = self.t
        for _ in range(round(seconds * self.fps)):
            self.t += 1 / self.fps
            self.clock.now += 1 / self.fps
            shown = hands(self.t - start) if callable(hands) else hands
            out += self.engine.update(Frame(self.t, list(shown), ASPECT))
        self.events += out
        return out


def kinds(events: Sequence[Event], kind: str) -> list[Event]:
    """The events of one type."""
    return [e for e in events if e["type"] == kind]


def names(events: Sequence[Event], phase: str | None = "start") -> list[str]:
    """The gestures reported, in order: their starts (phase=None: every report, hold steps included)."""
    return [e["gesture"] for e in events if e["type"] == "gesture" and (phase is None or e["phase"] == phase)]
