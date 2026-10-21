"""The gestures the tracker recognises: their names (shared with Orion), what each is called, and how to do it.

The tracker only reports gestures; what each one does is Orion's to decide (Settings → Gestures in his HUD). It never
sees Orion's actions, and Orion never sees video.

Three kinds:

- ``shape``: a hand shape held still. Reported once it has been held ``recognition.static_hold`` (``phase:
  start``), then again as it stays held (``phase: hold``) at :data:`HOLD_STEPS` seconds and every
  ``recognition.repeat_every`` after the last of them, so Orion can ask for a longer hold or repeat an action while the
  shape is held.
- ``motion``: a movement of one hand (a swipe, a wave, a circle, a push), reported once.
- ``two_hands``: both hands moving together, reported once.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

type Kind = Literal["shape", "motion", "two_hands"]


@dataclass(frozen=True, slots=True)
class Gesture:
    """One gesture of the vocabulary.

    Attributes:
        title: its name for people.
        kind: shape, motion or two_hands.
        how: how to do it, in one sentence.
    """

    title: str
    kind: Kind
    how: str


# MediaPipe's canned gestures, as the model names them, and the names used everywhere else.
CANNED: Final[Mapping[str, str]] = {
    "Open_Palm": "open_palm",
    "Closed_Fist": "closed_fist",
    "Pointing_Up": "pointing_up",
    "Thumb_Up": "thumb_up",
    "Thumb_Down": "thumb_down",
    "Victory": "victory",
    "ILoveYou": "i_love_you",
}

# In the order Orion lists them: one hand's shapes, its motions, then both hands.
GESTURES: Final[Mapping[str, Gesture]] = {
    "open_palm": Gesture("Open palm", "shape", "A flat hand, palm to the camera, fingers up."),
    "closed_fist": Gesture("Fist", "shape", "Close your hand into a fist, facing the camera."),
    "pointing_up": Gesture("Point up", "shape", "Only the index finger up."),
    "victory": Gesture("Victory", "shape", "Index and middle fingers up in a V."),
    "three": Gesture("Three", "shape", "Three fingers up: index, middle and ring (or thumb, index and middle)."),
    "thumb_up": Gesture("Thumb up", "shape", "A thumbs-up, the other fingers folded."),
    "thumb_down": Gesture("Thumb down", "shape", "A thumbs-down, the other fingers folded."),
    "i_love_you": Gesture("I love you", "shape", "Thumb, index and little finger out, middle and ring folded."),
    "call_me": Gesture("Call me", "shape", "Thumb and little finger out, the other three folded (a shaka)."),
    "ok_sign": Gesture("OK", "shape", "The thumb and index tips touching in a ring, the other three fingers up."),
    "pinch": Gesture("Pinch", "shape", "The thumb and index tips touching, the other fingers relaxed or folded."),
    "swipe_left": Gesture("Swipe left", "motion", "Sweep your hand to your left, about two palm lengths, quickly."),
    "swipe_right": Gesture("Swipe right", "motion", "Sweep your hand to your right, about two palm lengths, quickly."),
    "swipe_up": Gesture("Swipe up", "motion", "Sweep your hand upwards, about two palm lengths, quickly."),
    "swipe_down": Gesture("Swipe down", "motion", "Sweep your hand downwards, about two palm lengths, quickly."),
    "wave": Gesture("Wave", "motion", "Wave your open hand side to side, two or three times."),
    "circle_cw": Gesture("Circle clockwise", "motion", "Draw a circle in the air with your index finger, clockwise."),
    "circle_ccw": Gesture("Circle anticlockwise", "motion",
                          "Draw a circle in the air with your index finger, anticlockwise."),
    "push": Gesture("Push", "motion", "Push your open palm towards the camera, about a forearm's length."),
    "pull": Gesture("Pull", "motion", "Draw your open palm back from the camera, about a forearm's length."),
    "hands_apart": Gesture("Hands apart", "two_hands",
                           "Both hands up close together, then spread them apart within a second."),
    "hands_together": Gesture("Hands together", "two_hands",
                              "Both hands up apart, then bring them together within a second."),
}
SHAPES: Final[frozenset[str]] = frozenset(name for name, g in GESTURES.items() if g.kind == "shape")

# A shape still held is reported again at these seconds held, then every `recognition.repeat_every`.
HOLD_STEPS: Final[tuple[float, ...]] = (0.6, 1.0, 1.5, 2.0)


def describe() -> list[dict[str, str]]:
    """The vocabulary as JSON: one entry per gesture, in order (orion://gestures, and Orion's settings)."""
    return [{"name": name, "title": g.title, "kind": g.kind, "how": g.how} for name, g in GESTURES.items()]
