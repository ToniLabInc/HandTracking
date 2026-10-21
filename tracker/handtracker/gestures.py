"""From hands seen frame by frame to gestures: the tracker's logic, with no camera and no model.

Each processed frame brings up to two hands, each with MediaPipe's 21 landmarks, its handedness and the canned
gesture classifier's best guess. ``Engine.update`` turns a stream of such frames into events:

- ``gesture``: a gesture was made. A hand shape held still is reported once it has been held ``static_hold``
  (``phase: start``), then again while it stays held, at the hold steps (``phase: hold``, with how long it has been
  held), so Orion can wait for a longer hold or repeat an action. A motion of one hand or of both is reported once.
  What a gesture does is Orion's to decide: the tracker knows nothing of actions.
- ``presence``: the number of hands in view changed (debounced).

``Engine.view`` says, hand by hand, what the last frame showed (the live view's labels in Orion's HUD).

Coordinates are those of the mirrored image (as in a mirror: the user's right is the image's right), in units of the
image height, so x and y distances compare. Sizes and travel are measured in palm lengths (wrist to the middle
finger's knuckle, or the knuckle span when the hand is turned), which makes them independent of the distance to the
camera.

Hand shapes come from the canned classifier, and four from the landmarks themselves, which take precedence when they
are clear: ``ok_sign`` and ``pinch`` (the thumb and index tips touching), ``call_me`` and ``three`` (which fingers are
straight). A shape counts only while the hand is still, fires once per showing (then its hold steps), and the same
shape can't start again within ``cooldown``. Motions count only once a hand has been in view for ``settle``, so a hand
coming into view is not a swipe; a brief dropout of one of two hands is bridged, so both hands can come together.
"""

import itertools
import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from handtracker.config import TrackerConfig
from handtracker.vocabulary import CANNED, GESTURES, HOLD_STEPS, SHAPES

type Event = dict[str, Any]
type Point = tuple[float, float, float]
type Finger = Literal[1, 0, -1]  # straight, folded, unsure

TRACK_TIMEOUT_S: Final = 0.35  # a hand unseen this long is gone (its holds and motion are forgotten)
MATCH_MAX: Final = 3.0  # palm lengths: a hand farther than this from where a track was is a new hand
MIN_FRAMES: Final = 3  # a hold needs at least this many frames
SPEED_WINDOW_S: Final = 0.15  # stillness is judged over this window
HANDEDNESS_VOTES: Final = 15  # handedness is the majority of the last this many frames
KNUCKLE_SPAN: Final = 0.7  # index-to-pinky knuckle span / wrist-to-middle-knuckle length, on an average hand
COMPUTED_SCORE: Final = 0.9  # the confidence given to a shape read from the landmarks
FALLBACK_SCORE: Final = 0.75  # ...and to a canned shape the fingers make while the classifier is unsure
RISE: Final = 0.3  # palm lengths: a thumb this far above its knuckle points up (as far below: down)
POINT_RISE: Final = 0.5  # ...an index finger this far above its knuckle points up
UPRIGHT: Final = 1.0  # ...and a hand whose middle fingertip is this far above its wrist is held up, not lying down
OK_SPREAD: Final = 0.55  # palm lengths: an OK's index tip, curled into the ring, is this far from the middle tip
POINTING_REACH: Final = 0.55  # palm lengths: a pinch's index tip is at least this far from the palm's centre
CLOSE_FACTOR: Final = 1.4  # two hands this close (x together_distance) as one vanishes have come together
LINGER_S: Final = 0.4  # a wave or circle goes on while the hand moves: nothing new until it has rested this long
OPPOSITE: Final = {"swipe_left": "swipe_right", "swipe_right": "swipe_left", "swipe_up": "swipe_down",
                   "swipe_down": "swipe_up"}
FINGERS: Final = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))  # index..pinky: mcp, pip, dip, tip


@dataclass(frozen=True, slots=True)
class Hand:
    """One hand in one frame, as the recognizer saw it (mirrored image).

    Attributes:
        landmarks: 21 points (x, y, z): x and y normalised to the image width and height, z MediaPipe's relative depth
            (about the scale of x).
        handedness: "Left" or "Right" (the user's hand).
        gesture: the canned gesture classifier's best label ("Open_Palm", ..., "None").
        gesture_score: its confidence.
    """

    landmarks: Sequence[Point]
    handedness: str = "Right"
    gesture: str = "None"
    gesture_score: float = 0.0


@dataclass(frozen=True, slots=True)
class Frame:
    """One processed frame.

    Attributes:
        t: when it was captured, in monotonic seconds.
        hands: the hands seen (0, 1 or 2).
        aspect: the image's width / height.
    """

    t: float
    hands: Sequence[Hand] = ()
    aspect: float = 4 / 3


@dataclass(frozen=True, slots=True)
class HandView:
    """What the last frame showed of one hand, for the live view.

    Attributes:
        track: the hand's track (the same number while it stays in view).
        hand: "Left" or "Right" (the majority of the recent frames).
        shape: the hand shape it holds, None while it moves or holds none.
        score: how sure that shape is.
        held: how long it has held that shape, in seconds.
    """

    track: int
    hand: str
    shape: str | None
    score: float
    held: float


def _line(a: Point, b: Point, c: Point) -> float:
    """How straight the joint at b is: the cosine of the angle between a→b and b→c (1: in line)."""
    u, v = [q - p for p, q in zip(a, b, strict=True)], [q - p for p, q in zip(b, c, strict=True)]
    nu, nv = math.hypot(*u), math.hypot(*v)
    return math.sumprod(u, v) / (nu * nv) if nu > 1e-9 and nv > 1e-9 else 0.0


def fingers(pts: Sequence[Point], size: float) -> tuple[Finger, Finger, Finger, Finger, Finger]:
    """Which fingers are straight (1), folded (0) or unclear (-1): thumb, index, middle, ring, pinky.

    A finger is straight when its two segments line up and its tip is well beyond its middle joint, seen from the
    wrist; folded when its tip comes back towards the wrist, or bends right over. The thumb is straight when it is
    stretched away from the index knuckle, folded when it lies against the hand.
    """
    wrist = pts[0]
    out: list[Finger] = []
    thumb_away = math.dist(pts[4], pts[5]) / size
    thumb_out = _line(pts[2], pts[3], pts[4]) > 0.5 and math.dist(pts[4], pts[9]) / size > 0.7
    out.append(1 if thumb_away > 0.55 and thumb_out else 0 if thumb_away < 0.45 else -1)
    for mcp, pip, _dip, tip in FINGERS:
        reach = math.dist(pts[tip], wrist) / max(math.dist(pts[pip], wrist), 1e-6)
        line = _line(pts[mcp], pts[pip], pts[tip])
        out.append(1 if line > 0.75 and reach > 1.15 else 0 if reach < 1.0 or line < 0.2 else -1)
    return out[0], out[1], out[2], out[3], out[4]


@dataclass(slots=True)
class Observation:
    """What one hand in one frame amounts to."""

    cx: float  # palm centre, image-height units
    cy: float
    size: float  # palm length, image-height units
    handed: str
    canned: str | None  # canned shape (snake_case) confident enough to count
    score: float
    touch: float  # thumb tip to index tip, in palm lengths
    reach: float  # index tip to the palm's centre, in palm lengths
    tip: tuple[float, float]  # index tip, image-height units
    fingers: tuple[Finger, Finger, Finger, Finger, Finger]
    thumb_rise: float  # how far the thumb's tip is above its knuckle, in palm lengths (below: negative)
    index_rise: float  # ...and the index finger's
    spread: float  # the index tip to the middle finger's tip, in palm lengths (an OK's index curls away)
    upright: float  # how far the middle fingertip is above the wrist, in palm lengths (a hand held up)

    @property
    def open(self) -> bool:
        """Whether the hand is open: at least three of the four fingers straight, or the classifier's open palm."""
        return self.canned == "open_palm" or self.fingers[1:].count(1) >= 3


def observe(hand: Hand, aspect: float, min_score: float) -> Observation:
    """Measure one hand: palm centre and length, finger states, thumb-to-index and index reach, canned shape."""
    pts = [(x * aspect, y, z * aspect) for x, y, z in hand.landmarks]  # common units: the image height
    centre = [sum(c) / 5 for c in zip(*(pts[i] for i in (0, 5, 9, 13, 17)), strict=True)]
    size = max(math.dist(pts[0], pts[9]), math.dist(pts[5], pts[17]) / KNUCKLE_SPAN, 1e-6)
    canned = CANNED.get(hand.gesture) if hand.gesture_score >= min_score else None
    return Observation(centre[0], centre[1], size, hand.handedness, canned, float(hand.gesture_score),
                       math.dist(pts[4], pts[8]) / size, math.dist(pts[8], centre) / size, (pts[8][0], pts[8][1]),
                       fingers(pts, size), (pts[2][1] - pts[4][1]) / size, (pts[5][1] - pts[8][1]) / size,
                       math.dist(pts[8], pts[12]) / size, (pts[0][1] - pts[12][1]) / size)


def shape_of(o: Observation, touching: bool, ok_touch: float) -> tuple[str | None, float]:
    """The hand shape an observation shows, and how sure: a shape only the landmarks tell (OK, pinch, call me, three)
    when it is clear; else the canned classifier's; else, when the classifier is unsure, the canned shape the fingers
    make (it often says "None" for a hand seen at an angle that the fingers read plainly).

    Args:
        o: the observation.
        touching: whether the thumb and index tips touch (with the pinch's hysteresis).
        ok_touch: the OK sign's thumb and index may be this far apart (its ring is looser than a pinch).
    """
    thumb, index, middle, ring, pinky = o.fingers
    up = middle == ring == pinky == 1
    if o.canned != "closed_fist":
        if up and o.touch < ok_touch and o.spread >= OK_SPREAD:
            return "ok_sign", COMPUTED_SCORE
        if touching and not up and o.reach >= POINTING_REACH:
            return "pinch", COMPUTED_SCORE
    if thumb == 1 and pinky == 1 and index == middle == ring == 0:
        return "call_me", COMPUTED_SCORE
    if (index == middle == ring == 1 and pinky == 0 and thumb != 1) or \
            (thumb == index == middle == 1 and ring == pinky == 0 and o.canned != "i_love_you"):
        return "three", COMPUTED_SCORE
    if o.canned:
        return o.canned, o.score
    fallback = _fingered(o)
    return fallback, FALLBACK_SCORE if fallback else 0.0


def _fingered(o: Observation) -> str | None:
    """The canned shape the fingers make plainly, for when the classifier is unsure (None for any other).

    Only shapes a hand shows on purpose, held up to the camera: an open palm and a V with the fingers up, a finger
    pointing up, a thumb up or down, the I-love-you. Never a fist: a hand holding a mouse or a cup curls just the same
    (a fist is the classifier's alone).
    """
    thumb, index, middle, ring, pinky = o.fingers
    if index == middle == ring == pinky == 1 and thumb != 0 and o.upright >= UPRIGHT:
        return "open_palm"
    if index == middle == ring == pinky == 0:
        if thumb == 1 and abs(o.thumb_rise) >= RISE:
            return "thumb_up" if o.thumb_rise > 0 else "thumb_down"
        return None
    if index == middle == 1 and ring == pinky == 0 and thumb != 1 and o.index_rise >= POINT_RISE:
        return "victory"
    if index == 1 and middle == ring == pinky == 0 and thumb != 1 and o.index_rise >= POINT_RISE:
        return "pointing_up"
    if thumb == index == pinky == 1 and middle == ring == 0 and o.index_rise >= POINT_RISE:
        return "i_love_you"
    return None


@dataclass(slots=True)
class Sample:
    """One frame of a tracked hand, kept for holds and motions."""

    t: float
    cx: float
    cy: float
    size: float
    label: str | None  # the hand shape this frame (None while moving or unclear)
    score: float
    tip: tuple[float, float]
    open: bool


@dataclass(slots=True)
class Occurrence:
    """One showing of a hand shape by a tracked hand: from the first frame that shows it until it hasn't been seen
    for longer than ``max_gap``."""

    label: str
    start: float
    last_seen: float
    showing: int = 0  # the id of its start event, once reported
    next_hold: float = 0.0  # held this long, it is reported again
    quiet: bool = False  # never reported (ignored, or another hand showed it first)


@dataclass(slots=True)
class Track:
    """A hand followed from frame to frame."""

    id: int
    obs: Observation
    first_seen: float
    last_seen: float
    samples: deque[Sample] = field(default_factory=deque)
    votes: deque[str] = field(default_factory=lambda: deque(maxlen=HANDEDNESS_VOTES))
    touching: bool = False
    shapes: dict[str, Occurrence] = field(default_factory=dict)  # hand shapes being shown, by label

    @property
    def handed(self) -> str:
        """The handedness most recent frames agreed on."""
        return Counter(self.votes).most_common(1)[0][0] if self.votes else self.obs.handed


class Engine:
    """Turns frames into gesture and presence events (thread-safe).

    Args:
        config: the settings (the engine reads ``recognition`` and ``ignore``).
        wall: the clock for event timestamps and ids (unix seconds).
    """

    def __init__(self, config: TrackerConfig, *, wall: Callable[[], float] = time.time) -> None:
        self._lock = threading.RLock()
        self._wall = wall
        self.config = config
        self._tracks: list[Track] = []
        self._next_track = 1
        self._last_id = 0
        self._paused_until: float | None = None  # monotonic; math.inf: until resumed
        self._last_start: dict[str, float] = {}  # shape -> when it was last reported (its cooldown)
        self._motion_from = -math.inf  # samples older than this don't count for motions (cleared after one)
        self._swipe_block_until = -math.inf
        self._swipe_blocked: tuple[str, float] | None = None  # (direction, until): the hand coming back
        self._motion_block_until = -math.inf  # waves, circles, pushes and pulls
        self._lingering = False  # the last motion was a wave or a circle, which go on while the hand moves
        self._two_block_until = -math.inf
        self._two_up = -math.inf  # when two open hands were last up, or a two-hand gesture made: shapes are theirs
        self._pair: tuple[int, int] | None = None
        self._pair_samples: deque[tuple[float, float]] = deque()
        self.apart: float | None = None  # how far apart the two hands in view are, in palm lengths (the live view)
        self._presence = 0
        self._presence_pending: tuple[int, float] | None = None
        self._last_hand_t = -math.inf
        self.view: list[HandView] = []
        self.last_gesture: Event | None = None

    # ------------------------------------------------------------------------------------------------ control
    def configure(self, config: TrackerConfig) -> None:
        """Use new settings (from the next frame on)."""
        with self._lock:
            self.config = config

    def pause(self, t: float, seconds: float | None = None) -> None:
        """Report no gestures until ``resume``, or for ``seconds`` (the camera turns off meanwhile)."""
        with self._lock:
            self._paused_until = math.inf if seconds is None else t + seconds

    def resume(self) -> None:
        """Report gestures again."""
        with self._lock:
            self._paused_until = None

    def paused(self, t: float) -> bool:
        """Whether gestures are paused at monotonic time ``t`` (a timed pause ends by itself)."""
        with self._lock:
            if self._paused_until is not None and t >= self._paused_until:
                self._paused_until = None
            return self._paused_until is not None

    def rest(self) -> list[Event]:
        """The camera turned off: forget the hands in view (a ``presence`` event if there were any)."""
        with self._lock:
            self._tracks, self.view, self.apart = [], [], None
            self._pair, self._pair_samples, self._presence_pending = None, deque(), None
            if not self._presence:
                return []
            self._presence = 0
            return [{"type": "presence", "hands": 0, "ts": self._ts()}]

    def hand_seen_within(self, t: float, seconds: float) -> bool:
        """Whether a hand was in view in the last ``seconds`` (the camera loop slows down when none was)."""
        return t - self._last_hand_t <= seconds

    def snapshot(self, t: float) -> dict[str, Any]:
        """The state, for /state and the status events."""
        with self._lock:
            paused = self.paused(t)
            until = self._paused_until
            return {
                "hands": self._presence,
                "paused": paused,
                "paused_for": None if until is None or until == math.inf else round(until - t, 1),
                "last_gesture": self.last_gesture,
            }

    # ------------------------------------------------------------------------------------------------ frames
    def update(self, frame: Frame) -> list[Event]:
        """Take one processed frame; return the events it caused, in order."""
        with self._lock:
            t = frame.t
            s = self.config.recognition
            obs = [observe(h, frame.aspect, s.min_score) for h in frame.hands[:2]]
            if obs:
                self._last_hand_t = t
            seen = self._match(obs, t)
            for track in seen:
                self._sample(track, t)
            if self._lingering and t >= self._motion_block_until:
                self._lingering = False  # the wave or circle is over, hands in view or not
            if len(seen) == 2 and all(tr.samples[-1].open for tr in seen):
                self._two_up = t  # both hands up and open: the two-hand gestures' pose, not one hand's shape
            events: list[Event] = []
            for track in seen:
                events += self._static(track, t)
            events += self._two_hands(t)
            for track in seen:
                if self._others_move(track, seen, t):
                    continue  # both hands moving: a two-hand gesture, not one hand's motion
                self._linger(track, t)
                events += self._swipe(track, t) or self._wave(track, t) or self._circle(track, t) \
                    or self._push(track, t)
            events += self._presence_update(len(obs), t)
            self.view = [self._view(track, t) for track in seen]
            if self.paused(t):
                events = [e for e in events if e["type"] != "gesture"]
            return events

    def inject(self, gesture: str) -> list[Event]:
        """Act as if ``gesture`` was just made (a shape: its start), by the "debug" hand. For debugging and tests."""
        with self._lock:
            event = self._event(gesture, "start", "debug", 1.0)
            if gesture in SHAPES:
                event["showing"] = event["id"]
            return [event]

    # ------------------------------------------------------------------------------------------------ tracking
    def _keep(self) -> float:
        """How long an unseen hand's track lasts: TRACK_TIMEOUT_S, or longer when two_hand_gap asks for it (so a hand
        of a pair that comes back is the same hand)."""
        return max(TRACK_TIMEOUT_S, self.config.recognition.two_hand_gap)

    def _match(self, obs: list[Observation], t: float) -> list[Track]:
        """Pair this frame's hands with the tracked ones (the most pairs, then the nearest palm centres); a new hand
        starts a new track, and a track unseen too long is dropped."""
        live = self._tracks = [tr for tr in self._tracks if t - tr.last_seen <= self._keep()]

        def cost(o: Observation, tr: Track) -> float:
            d = math.hypot(o.cx - tr.obs.cx, o.cy - tr.obs.cy) / max(o.size, tr.obs.size)
            return d + 0.25 if tr.handed != o.handed else d  # a weak preference for the same handedness

        options = [[(oi, ti, d) for oi, ti in option if (d := cost(obs[oi], live[ti])) <= MATCH_MAX]
                   for option in _assignments(len(obs), len(live))]
        best = min(options, key=lambda pairs: (-len(pairs), sum(d for _, _, d in pairs)))
        chosen = {oi: live[ti] for oi, ti, _ in best}
        seen = []
        for oi, o in enumerate(obs):
            track = chosen.get(oi)
            if track is None:
                track = Track(self._next_track, o, t, t)
                self._next_track += 1
                self._tracks.append(track)
            track.obs, track.last_seen = o, t
            track.votes.append(o.handed)
            seen.append(track)
        return seen

    def _sample(self, track: Track, t: float) -> None:
        """Record this frame of a track: its hand shape, unless the hand is moving."""
        s, o = self.config.recognition, track.obs
        track.touching = o.touch < (s.pinch_off if track.touching else s.pinch_on)
        label, score = shape_of(o, track.touching, s.ok_touch)
        if label is not None and self._speed(track, t) > s.still_speed:
            label = None
        track.samples.append(Sample(t, o.cx, o.cy, o.size, label, score, o.tip, o.open))
        keep = max(s.static_hold, s.swipe_max_time, s.wave_window, s.circle_window, s.push_window,
                   SPEED_WINDOW_S) + 0.5
        while track.samples and t - track.samples[0].t > keep:
            track.samples.popleft()

    def _speed(self, track: Track, t: float) -> float:
        """How fast the palm moves, in palm lengths per second, over the last ``SPEED_WINDOW_S``."""
        o = track.obs
        for sample in track.samples:  # oldest first: the oldest one inside the window
            dt = t - sample.t
            if dt <= SPEED_WINDOW_S:
                return math.hypot(o.cx - sample.cx, o.cy - sample.cy) / o.size / dt if dt > 0 else 0.0
        return 0.0

    def _others_move(self, track: Track, seen: list[Track], t: float) -> bool:
        """Whether another hand in view moves too."""
        s = self.config.recognition
        return any(self._speed(other, t) > s.still_speed for other in seen if other is not track)

    def _view(self, track: Track, t: float) -> HandView:
        """One hand as the live view shows it."""
        sample = track.samples[-1]
        occ = track.shapes.get(sample.label) if sample.label else None
        return HandView(track.id, track.handed, sample.label, round(sample.score, 2),
                        round(t - occ.start, 2) if occ is not None else 0.0)

    # ------------------------------------------------------------------------------------------------ hand shapes
    def _held(self, track: Track, occ: Occurrence, hold: float, t: float) -> float | None:
        """Whether a hand shape has been held for ``hold`` seconds up to now: its mean score if so, else None.

        Held means: this frame shows it; it was first shown at least ``hold`` ago; the frames of the last ``hold``
        seconds have no gap longer than ``max_gap``, number at least ``MIN_FRAMES``, and show it in at least
        ``hold_ratio`` of them.
        """
        s = self.config.recognition
        samples = track.samples
        if not samples or samples[-1].label != occ.label or t - occ.start < hold - 1e-6:
            return None
        start = t - hold - 1e-6
        n = hits = 0
        total = 0.0
        newer = t
        for sample in reversed(samples):
            if sample.t < start:
                break
            if newer - sample.t > s.max_gap:
                return None
            newer = sample.t
            n += 1
            if sample.label == occ.label:
                hits += 1
                total += sample.score
        if n < MIN_FRAMES or hits < s.hold_ratio * n:
            return None
        return total / hits

    def _static(self, track: Track, t: float) -> list[Event]:
        """Hand shapes: report one once held still ``static_hold`` (its start), then at each hold step it reaches."""
        s = self.config.recognition
        label = track.samples[-1].label
        for gone in [k for k, occ in track.shapes.items() if t - occ.last_seen > s.max_gap + 1e-6]:
            del track.shapes[gone]  # not seen for longer than a dropout: that showing ended
        two_up = t - self._two_up <= s.two_hand_quiet + 1e-6
        if two_up:
            for occ in track.shapes.values():
                occ.quiet = True  # under way as both hands came up: part of a two-hand gesture, never reported
        if label is None:
            return []
        occ = track.shapes.get(label)
        if occ is None:
            occ = track.shapes[label] = Occurrence(label, t, t, quiet=two_up)
        occ.last_seen = t
        if occ.quiet:
            return []
        held = t - occ.start
        if occ.showing and (held + 1e-6 < occ.next_hold or occ.next_hold > s.max_hold + 1e-6):
            return []  # reported already: not at its next hold step yet, or held long enough
        score = self._held(track, occ, s.static_hold, t)
        if score is None:
            return []
        if not occ.showing:
            if label in self.config.ignore or any(other is not track and (o := other.shapes.get(label)) is not None
                                               and o.showing and not o.quiet for other in self._tracks):
                occ.quiet = True  # kept quiet, or another hand shows it already: once is enough
                return []
            if t - self._last_start.get(label, -math.inf) < s.cooldown:
                return []  # shown again too soon: it waits for the cooldown (still held, it starts then)
            self._last_start[label] = t
        event = self._event(label, "hold" if occ.showing else "start", track.handed, score, held=held)
        event["showing"] = occ.showing = occ.showing or event["id"]
        occ.next_hold = self._next_step(held)  # past any step this frame already went beyond: one report, not two
        return [event]

    def _next_step(self, held: float) -> float:
        """The first hold step after ``held``: HOLD_STEPS, then every ``repeat_every`` after the last of them."""
        for step in HOLD_STEPS:
            if step > held + 1e-6:
                return step
        every, last = self.config.recognition.repeat_every, HOLD_STEPS[-1]
        return last + (math.floor((held - last) / every + 1e-6) + 1) * every

    # ------------------------------------------------------------------------------------------------ motions
    def _recent(self, track: Track, t: float, window: float) -> list[Sample]:
        """A track's samples of the last ``window`` that may count for a motion: after the hand settled in view and
        after the last motion."""
        s = self.config.recognition
        since = max(t - window, track.first_seen + s.settle, self._motion_from)
        return [sample for sample in track.samples if sample.t >= since - 1e-9]

    def _swipe(self, track: Track, t: float) -> list[Event]:
        """A swipe: the palm travelled far enough, straight enough, within the swipe window."""
        s = self.config.recognition
        if t < self._swipe_block_until:
            return []
        cur = track.samples[-1]
        best: tuple[float, str, float] | None = None  # (travel, direction, travel in palm lengths)
        for sample in self._recent(track, t, s.swipe_max_time):
            if t - sample.t < s.swipe_min_time:
                break
            dx, dy = cur.cx - sample.cx, cur.cy - sample.cy
            if abs(dx) >= abs(dy):
                along, across, direction = abs(dx), abs(dy), "swipe_right" if dx > 0 else "swipe_left"
            else:
                along, across, direction = abs(dy), abs(dx), "swipe_down" if dy > 0 else "swipe_up"
            palms = along / ((cur.size + sample.size) / 2)
            if palms < s.swipe_distance or along < s.swipe_min_travel or along < s.swipe_straightness * across:
                continue
            if best is None or along > best[0]:
                best = (along, direction, palms)
        if best is None:
            return []
        _, direction, palms = best
        if self._swipe_blocked is not None and self._swipe_blocked[0] == direction and t < self._swipe_blocked[1]:
            return []
        self._sweep_done(t)
        self._swipe_blocked = (OPPOSITE[direction], t + s.swipe_return_block)
        return self._motion(direction, track.handed, min(1.0, palms / (2 * s.swipe_distance)))

    def _wave(self, track: Track, t: float) -> list[Event]:
        """A wave: an open hand swinging side to side, ``wave_swings`` times, each swing neither tiny nor a swipe."""
        s = self.config.recognition
        if t < self._motion_block_until:
            return []
        samples = self._recent(track, t, s.wave_window)
        if len(samples) < 6 or sum(1 for x in samples if x.open) < 0.6 * len(samples):
            return []
        size = sum(x.size for x in samples) / len(samples)
        if (max(x.cy for x in samples) - min(x.cy for x in samples)) / size > 1.0:
            return []  # it wandered up or down: not a wave
        legs = _swings([x.cx / size for x in samples], s.wave_min_swing)
        if any(leg > s.wave_max_swing for leg in legs) or len(legs) < s.wave_swings:
            return []  # too few swings, or one as wide as a swipe
        self._motion_done(t, lingers=True)
        return self._motion("wave", track.handed, min(1.0, len(legs) / (s.wave_swings + 1)))

    def _circle(self, track: Track, t: float) -> list[Event]:
        """A circle: the index tip turned round its path's centre by ``circle_turn`` degrees, one way."""
        s = self.config.recognition
        if t < self._motion_block_until:
            return []
        samples = self._recent(track, t, s.circle_window)
        if len(samples) < 8:
            return []
        cx = sum(x.tip[0] for x in samples) / len(samples)
        cy = sum(x.tip[1] for x in samples) / len(samples)
        radii = [math.hypot(x.tip[0] - cx, x.tip[1] - cy) for x in samples]
        mean = sum(radii) / len(radii)
        size = sum(x.size for x in samples) / len(samples)
        if mean / size < s.circle_min_radius:
            return []
        if math.sqrt(sum((r - mean) ** 2 for r in radii) / len(radii)) > 0.4 * mean:
            return []  # not round enough (a line back and forth isn't)
        ways = [0.0, 0.0]  # degrees turned clockwise (image y points down: positive), and anticlockwise
        previous: float | None = None
        for x, r in zip(samples, radii, strict=True):
            if r < 0.4 * mean:
                continue  # too near the centre for its angle to mean anything
            angle = math.atan2(x.tip[1] - cy, x.tip[0] - cx)
            if previous is not None:
                step = math.degrees((angle - previous + math.pi) % (2 * math.pi) - math.pi)
                if abs(step) < 120:  # a bigger jump is noise, not turning
                    ways[step < 0] += abs(step)
            previous = angle
        degrees = ways[0] - ways[1]
        if abs(degrees) < s.circle_turn or min(ways) > 0.25 * max(ways):
            return []  # not far enough round, or not one way round
        self._motion_done(t, lingers=True)
        return self._motion("circle_cw" if degrees > 0 else "circle_ccw", track.handed,
                            min(1.0, abs(degrees) / 360))

    def _push(self, track: Track, t: float) -> list[Event]:
        """A push or a pull: an open palm growing or shrinking by ``push_scale`` without moving much sideways."""
        s = self.config.recognition
        if t < self._motion_block_until or not track.samples[-1].open:
            return []
        samples = self._recent(track, t, s.push_window)
        if len(samples) < 5:
            return []
        cur = samples[-1]
        now = sorted(x.size for x in samples[-3:])[1]  # the median of the last three: one odd frame is no push
        early = samples[:-3]
        small = min(early, key=lambda x: x.size)
        big = max(early, key=lambda x: x.size)
        if now / small.size >= s.push_scale and math.hypot(cur.cx - small.cx, cur.cy - small.cy) / now <= 1.0:
            gesture, ratio = "push", now / small.size
        elif big.size / now >= s.push_scale and math.hypot(cur.cx - big.cx, cur.cy - big.cy) / big.size <= 1.0:
            gesture, ratio = "pull", big.size / now
        else:
            return []
        self._motion_done(t)
        return self._motion(gesture, track.handed, min(1.0, (ratio - 1) / (2 * (s.push_scale - 1))))

    def _motion_done(self, t: float, *, lingers: bool = False) -> None:
        """A wave, circle, push or pull was recognised: no other motion counts for a while (for a wave or a circle,
        until the hand has rested too)."""
        s = self.config.recognition
        self._motion_block_until = t + s.motion_cooldown
        self._swipe_block_until = max(self._swipe_block_until, t + s.swipe_cooldown)
        self._motion_from, self._lingering = t, lingers

    def _sweep_done(self, t: float) -> None:
        """A swipe or a two-hand gesture was recognised: no motion counts for ``swipe_cooldown`` (what moves
        meanwhile doesn't count after it either), and a wave or circle under way is over."""
        s = self.config.recognition
        self._swipe_block_until = self._motion_from = max(self._swipe_block_until, t + s.swipe_cooldown)
        self._motion_block_until = max(self._motion_block_until, t + s.swipe_cooldown)
        self._lingering = False

    def _linger(self, track: Track, t: float) -> None:
        """A wave or circle goes on while the hand keeps moving: it isn't reported again until the hand rests."""
        if not self._lingering:
            return
        if t >= self._motion_block_until:
            self._lingering = False  # it rested: over
        elif self._speed(track, t) > self.config.recognition.still_speed / 2:
            self._motion_block_until = max(self._motion_block_until, t + LINGER_S)
            self._motion_from = t

    def _two_hands(self, t: float) -> list[Event]:
        """hands_apart / hands_together: within the window, the palms went from together to apart, or back.

        One of the two may drop out of view for ``two_hand_gap`` (its last place stands in), and two hands closing
        in fast often merge into one for the recognizer: one vanishing while they are close counts as together.
        """
        s = self.config.recognition
        live = sorted((tr for tr in self._tracks if t - tr.last_seen <= s.two_hand_gap),
                      key=lambda tr: tr.last_seen, reverse=True)[:2]
        samples = self._pair_samples
        while samples and t - samples[0][0] > s.two_hand_window:
            samples.popleft()
        if len(live) < 2:
            self.apart = None
            if self._pair is not None and samples and t >= self._two_block_until:
                closing = len(samples) >= 2 and samples[-1][1] < samples[0][1]
                if closing and samples[-1][1] <= CLOSE_FACTOR * s.together_distance and \
                        max(d for _, d in samples) >= s.apart_distance:
                    return self._two_done("hands_together", samples, t)
            self._pair, self._pair_samples = None, deque()
            return []
        a, b = sorted(live, key=lambda tr: tr.id)
        if self._pair != (a.id, b.id):
            self._pair, self._pair_samples = (a.id, b.id), deque()
            samples = self._pair_samples
        d = math.hypot(a.obs.cx - b.obs.cx, a.obs.cy - b.obs.cy) / ((a.obs.size + b.obs.size) / 2)
        samples.append((t, d))
        self.apart = d
        if t < self._two_block_until:
            return []
        if d >= s.apart_distance and any(old <= s.together_distance for _, old in samples):
            return self._two_done("hands_apart", samples, t)
        if d <= s.together_distance and any(old >= s.apart_distance for _, old in samples):
            return self._two_done("hands_together", samples, t)
        return []

    def _two_done(self, gesture: str, samples: deque[tuple[float, float]], t: float) -> list[Event]:
        """A two-hand gesture was recognised: report it, and let nothing else move for a moment."""
        s = self.config.recognition
        span = max(d for _, d in samples) - min(d for _, d in samples)
        samples.clear()
        self._two_block_until = t + s.two_hand_cooldown
        self._two_up = t
        self._sweep_done(t)
        return self._motion(gesture, "both", min(1.0, span / (2 * (s.apart_distance - s.together_distance))))

    def _motion(self, gesture: str, hand: str, score: float) -> list[Event]:
        """A motion was recognised: its event, unless it is kept quiet."""
        if gesture in self.config.ignore:
            return []
        return [self._event(gesture, "start", hand, score)]

    # ------------------------------------------------------------------------------------------------ events
    def _event(self, gesture: str, phase: str, hand: str, score: float, *, held: float = 0.0) -> Event:
        """A ``gesture`` event with the next id (ids are milliseconds-based: they increase even across restarts)."""
        self._last_id = max(self._last_id + 1, int(self._wall() * 1000))
        event: Event = {"type": "gesture", "id": self._last_id, "gesture": gesture, "kind": GESTURES[gesture].kind,
                        "phase": phase, "hand": hand, "score": round(score, 2), "held": round(held, 2),
                        "ts": self._ts()}
        if phase == "start":
            self.last_gesture = event
        return event

    def _presence_update(self, count: int, t: float) -> list[Event]:
        """Report a change in the number of hands once it has lasted ``presence_debounce``."""
        if count == self._presence:
            self._presence_pending = None
            return []
        if self._presence_pending is None or self._presence_pending[0] != count:
            self._presence_pending = (count, t)
        if t - self._presence_pending[1] < self.config.recognition.presence_debounce:
            return []
        self._presence, self._presence_pending = count, None
        return [{"type": "presence", "hands": count, "ts": self._ts()}]

    def _ts(self) -> float:
        """Now, as a unix timestamp."""
        return round(self._wall(), 3)


def _swings(xs: Sequence[float], least: float) -> list[float]:
    """The swings of a back-and-forth movement: the distance between successive turning points (the last one under
    way included). A turn counts only once the movement has come back by ``least``, so jitter is no swing.

    Args:
        xs: positions, oldest first.
        least: the smallest swing.

    Returns:
        Each swing's length, oldest first.
    """
    if not xs:
        return []
    pivots, extreme, direction = [xs[0]], xs[0], 0
    for x in xs[1:]:
        if direction == 0:
            if abs(x - pivots[-1]) >= least:
                direction, extreme = (1 if x > pivots[-1] else -1), x
        elif (x - extreme) * direction >= 0:
            extreme = x  # still going the same way
        elif abs(x - extreme) >= least:
            pivots.append(extreme)  # it came back far enough: that was a turning point
            direction, extreme = -direction, x
    legs = [abs(b - a) for a, b in itertools.pairwise(pivots)]
    if direction != 0 and abs(extreme - pivots[-1]) >= least:
        legs.append(abs(extreme - pivots[-1]))
    return legs


def _assignments(n_obs: int, n_tracks: int) -> list[list[tuple[int, int]]]:
    """Every way to pair the observations (at most two) with the live tracks, each used at most once (without either:
    one way, no pairs)."""
    k = min(n_obs, n_tracks)
    return [list(zip(chosen, tracks, strict=True)) for chosen in itertools.combinations(range(n_obs), k)
            for tracks in itertools.permutations(range(n_tracks), k)]
