"""The gesture logic on synthetic landmark sequences: shapes and their holds, the shapes read from the landmarks,
swipes, waves, circles, pushes, two hands, presence, pausing, and the recognizer's scheduler."""

import dataclasses
import math
from types import SimpleNamespace as NS

import pytest

from handtracker import config
from handtracker.config import TrackerConfig
from handtracker.gestures import Engine, Frame, fingers, observe, shape_of
from handtracker.vision import Scheduler, hands_from_result
from tests.synth import ASPECT, Clock, Sim, hand, kinds, names


def tuned(cfg: TrackerConfig, **recognition: float) -> TrackerConfig:
    """The settings with some recognition values changed."""
    return cfg.model_copy(update={"recognition": cfg.recognition.model_copy(update=recognition)})


PALM = hand(0.66, 0.5, gesture="Open_Palm")
FIST = hand(0.66, 0.5, gesture="Closed_Fist", fingers="00000")
VICTORY = hand(0.66, 0.5, gesture="Victory", fingers="01100")
THUMB_UP = hand(0.66, 0.5, gesture="Thumb_Up", fingers="10000")


def starts(events: list) -> list:
    """The start events of the gestures reported."""
    return [e for e in kinds(events, "gesture") if e["phase"] == "start"]


def moves(events: list) -> list[str]:
    """The motions reported (one hand's and both hands'), in order: the hand shapes at rest left out."""
    return [e["gesture"] for e in starts(events) if e["kind"] != "shape"]


# ---------------------------------------------------------------------------------------------------- hand shapes
def test_a_hand_shape_is_reported_once_held_still(sim: Sim) -> None:
    assert kinds(sim.run(0.25, [VICTORY]), "gesture") == []  # 5 frames: not held long enough yet
    events = sim.run(0.1, [VICTORY])
    assert names(events) == ["victory"]
    event = events[0]
    assert event["hand"] == "Right" and event["score"] == 0.9 and event["kind"] == "shape"
    assert event["phase"] == "start" and event["showing"] == event["id"] and 0.25 <= event["held"] <= 0.35
    assert names(sim.run(0.2, [VICTORY])) == []  # once per showing


def test_a_shape_held_on_is_reported_at_its_hold_steps(sim: Sim) -> None:
    events = kinds(sim.run(3.05, [THUMB_UP]), "gesture")
    assert [e["phase"] for e in events][:2] == ["start", "hold"]
    holds = [e["held"] for e in events if e["phase"] == "hold"]
    assert holds[:4] == pytest.approx([0.6, 1.0, 1.5, 2.0], abs=0.06)  # the hold steps
    assert holds[4:] == pytest.approx([2.5, 3.0], abs=0.06)  # then every repeat_every
    assert {e["showing"] for e in events} == {events[0]["id"]}  # all one showing
    assert len({e["id"] for e in events}) == len(events)


def test_holding_stops_being_reported_after_max_hold(sim: Sim, cfg: TrackerConfig) -> None:
    sim.engine.configure(tuned(cfg, max_hold=1.2))
    events = kinds(sim.run(3.0, [FIST]), "gesture")
    assert [round(e["held"], 1) for e in events] == [0.3, 0.6, 1.0]


def test_a_flickering_classifier_still_counts_as_held(sim: Sim) -> None:
    unsure = hand(0.66, 0.5, gesture="Victory", fingers="01100", score=0.3)  # below min_score: no label this frame
    events = sim.run(1.2, lambda t: [unsure] if round(t * 20) % 5 == 0 else [VICTORY])
    assert names(events) == ["victory"]


def test_a_hand_that_leaves_ends_the_showing(sim: Sim) -> None:
    assert names(sim.run(0.5, [VICTORY])) == ["victory"]
    sim.run(1.2)  # gone longer than a dropout, and longer than the cooldown
    assert names(sim.run(0.5, [VICTORY])) == ["victory"]  # a new showing


def test_a_short_dropout_doesnt_end_the_showing(sim: Sim) -> None:
    events = sim.run(2.0, lambda t: [] if 0.8 < t < 0.95 else [VICTORY])
    assert names(events) == ["victory"]
    assert len(names(events, "hold")) >= 3  # the hold went on through it


def test_a_moving_hand_doesnt_hold_a_shape(sim: Sim) -> None:
    events = sim.run(1.5, lambda t: [hand(0.3 + 0.4 * t, 0.5, gesture="Victory", fingers="01100")])  # 2.7 palms/s
    assert "victory" not in names(events)


def test_the_same_shape_waits_out_its_cooldown(sim: Sim) -> None:
    first = starts(sim.run(0.4, [THUMB_UP]))
    sim.run(0.4)  # let go (longer than a forgiven dropout)
    second = starts(sim.run(1.5, [THUMB_UP]))  # shown again at once: it starts once the cooldown is over
    assert len(first) == len(second) == 1
    assert second[0]["ts"] - first[0]["ts"] >= sim.engine.config.recognition.cooldown - 1e-6


def test_two_hands_with_the_same_shape_report_it_once(sim: Sim) -> None:
    other = hand(0.2, 0.5, gesture="Victory", fingers="01100", handedness="Left")
    events = sim.run(3.0, [VICTORY, other])
    assert names(events) == ["victory"]


def test_gesture_ids_increase(sim: Sim) -> None:
    sim.run(0.5, [VICTORY])
    sim.run(1.5)
    sim.run(0.5, [THUMB_UP])
    ids = [e["id"] for e in kinds(sim.events, "gesture")]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_shapes_kept_quiet_are_never_reported(sim: Sim, cfg: TrackerConfig) -> None:
    sim.engine.configure(cfg.model_copy(update={"ignore": frozenset({"victory"})}))
    assert kinds(sim.run(2.0, [VICTORY]), "gesture") == []
    sim.run(1.0)
    assert names(sim.run(0.5, [THUMB_UP])) == ["thumb_up"]


# ---------------------------------------------------------------------------------------------- read from the landmarks
@pytest.mark.parametrize(("kw", "expected"), [
    ({"fingers": "01110"}, "three"),  # index, middle, ring
    ({"fingers": "11100"}, "three"),  # thumb, index, middle
    ({"fingers": "10001"}, "call_me"),  # thumb and little finger
    ({"fingers": "11111", "ok": True}, "ok_sign"),
    ({"fingers": "11000", "pinch": True}, "pinch"),
    ({"fingers": "11001", "gesture": "ILoveYou"}, "i_love_you"),  # not three: the classifier's own
])
def test_shapes_read_from_the_landmarks(sim: Sim, kw: dict, expected: str) -> None:
    assert names(sim.run(0.5, [hand(0.66, 0.5, **kw)])) == [expected]


@pytest.mark.parametrize(("fingers", "expected"), [
    ("11111", "open_palm"), ("01100", "victory"), ("01000", "pointing_up"), ("11001", "i_love_you"),
    ("10000", "thumb_up"), ("01010", None),
    ("00000", None),  # never a fist: a hand round a mouse or a cup curls the same; a fist is the classifier's alone
])
def test_an_unsure_classifier_leaves_the_shape_to_the_fingers(fingers: str, expected: str | None) -> None:
    o = observe(hand(0.66, 0.5, fingers=fingers, gesture="None", score=0.93), ASPECT, 0.5)
    assert shape_of(o, False, 0.42) == (expected, 0.75 if expected else 0.0)


def test_a_hand_lying_down_is_no_open_palm() -> None:
    lying = hand(0.66, 0.5, gesture="None")
    flat = dataclasses.replace(lying, landmarks=[(0.5 + (y - 0.5), 0.5 + (x - 0.5) * 0.1, z)
                                                 for x, y, z in lying.landmarks])  # turned on its side, flattened
    assert shape_of(observe(flat, ASPECT, 0.5), False, 0.42)[0] is None


def test_a_confident_classifier_is_believed() -> None:
    o = observe(hand(0.66, 0.5, fingers="01100", gesture="Pointing_Up", score=0.8), ASPECT, 0.5)
    assert shape_of(o, False, 0.42) == ("pointing_up", 0.8)


def test_the_ok_signs_ring_may_be_looser_than_a_pinch() -> None:
    o = observe(hand(0.66, 0.5, fingers="11111", ok=True), ASPECT, 0.5)
    assert shape_of(o, False, 0.42)[0] == "ok_sign"  # not touching as a pinch, still an OK


def test_finger_states() -> None:
    o = observe(hand(0.66, 0.5, fingers="10101"), ASPECT, 0.5)
    assert o.fingers == (1, 0, 1, 0, 1)
    pts = [(x * ASPECT, y, z) for x, y, z in hand(0.66, 0.5).landmarks]
    assert fingers(pts, 0.15) == (1, 1, 1, 1, 1)


def test_a_curled_hand_is_no_pinch(sim: Sim) -> None:
    # the thumb resting on a curled index: touching, but the index tip is in the palm, not out in a pinch
    curled = hand(0.66, 0.5, fingers="00000")
    assert observe(curled, ASPECT, 0.5).touch < 0.22
    assert "pinch" not in names(sim.run(1.0, [curled]))


def test_a_fist_is_never_a_pinch(sim: Sim) -> None:
    assert names(sim.run(0.5, [FIST])) == ["closed_fist"]


def test_pinch_has_hysteresis(sim: Sim) -> None:
    def near(gap: float):
        h = hand(0.66, 0.5, fingers="11000", pinch=True)
        pts = list(h.landmarks)
        x, y, z = pts[8]
        pts[4] = (x + gap * 0.15 / ASPECT, y, z)
        return dataclasses.replace(h, landmarks=pts)

    events = sim.run(0.5, [near(0.1)])  # touching
    assert names(events) == ["pinch"]
    events = sim.run(0.5, [near(0.3)])  # between the two thresholds: still a pinch (no new showing)
    assert names(events) == []
    assert "pinch" in {v.shape for v in sim.engine.view}


# ---------------------------------------------------------------------------------------------------- swipes
def swipe(dx: float, dy: float, seconds: float, *, size: float = 0.15, gesture: str = "None"):
    """A hand at rest 0.4 s, then moving by (dx, dy) image-height units over ``seconds``, then at rest."""

    def at(t: float):
        k = min(max((t - 0.4) / seconds, 0.0), 1.0)
        return [hand(0.66 - dx / 2 + dx * k, 0.5 - dy / 2 + dy * k, size=size, gesture=gesture)]

    return at


@pytest.mark.parametrize(("dx", "dy", "expected"), [
    (0.5, 0.0, "swipe_right"),  # towards the user's right (the mirrored image's right)
    (-0.5, 0.0, "swipe_left"),
    (0.0, -0.45, "swipe_up"),
    (0.0, 0.45, "swipe_down"),
])
def test_swipes_in_four_directions(sim: Sim, dx: float, dy: float, expected: str) -> None:
    events = [e for e in starts(sim.run(1.5, swipe(dx, dy, 0.3))) if e["kind"] == "motion"]
    assert [e["gesture"] for e in events] == [expected]
    assert events[0]["hand"] == "Right" and 0 < events[0]["score"] <= 1


def test_an_open_palm_swipe_is_only_a_swipe(sim: Sim) -> None:
    assert names(sim.run(1.2, swipe(0.5, 0.0, 0.3, gesture="Open_Palm"))) == ["open_palm", "swipe_right"]


def test_a_slow_drift_is_not_a_swipe(sim: Sim) -> None:
    assert moves(sim.run(3.0, swipe(0.5, 0.0, 2.0))) == []


def test_a_short_twitch_is_not_a_swipe(sim: Sim) -> None:
    assert moves(sim.run(1.5, swipe(0.15, 0.0, 0.2))) == []  # one palm length


def test_a_diagonal_is_not_a_swipe(sim: Sim) -> None:
    assert moves(sim.run(1.5, swipe(0.4, 0.35, 0.3))) == []


def test_a_far_hand_swipes_by_palm_lengths(sim: Sim) -> None:
    assert moves(sim.run(1.5, swipe(0.2, 0.0, 0.3, size=0.07))) == ["swipe_right"]  # 2.9 palm lengths


def test_a_hand_coming_into_view_is_not_a_swipe(sim: Sim) -> None:
    # it appears at the edge already moving, and sweeps in: the motion before it settled doesn't count
    assert moves(sim.run(1.0, lambda t: [hand(0.1 + 1.6 * min(t, 0.3), 0.5)])) == []


def test_the_hand_coming_back_is_not_a_swipe(sim: Sim) -> None:
    def there_and_back(t: float):
        if t < 0.4:
            x = 0.4
        elif t < 0.7:
            x = 0.4 + 0.5 * (t - 0.4) / 0.3
        elif t < 1.0:
            x = 0.9 - 0.5 * (t - 0.7) / 0.3
        else:
            x = 0.4
        return [hand(x, 0.5)]

    assert moves(sim.run(2.0, there_and_back)) == ["swipe_right"]


def test_one_motion_swipes_once(sim: Sim) -> None:
    assert moves(sim.run(2.0, swipe(0.9, 0.0, 0.5))) == ["swipe_right"]


def test_swipes_kept_quiet_still_block_the_hand_coming_back(sim: Sim, cfg: TrackerConfig) -> None:
    sim.engine.configure(cfg.model_copy(update={"ignore": frozenset({"swipe_right"})}))
    assert moves(sim.run(2.0, swipe(0.5, 0.0, 0.3))) == []


# ---------------------------------------------------------------------------------------------------- other motions
def wave(swing: float, hz: float, *, size: float = 0.15, gesture: str = "Open_Palm", fingers: str = "11111"):
    """An open hand at rest 0.4 s, then swinging side to side: ``swing`` palm lengths from side to side."""

    def at(t: float):
        k = max(t - 0.4, 0.0)
        return [hand(0.66 + swing / 2 * size * math.sin(2 * math.pi * hz * k), 0.5, size=size, gesture=gesture,
                     fingers=fingers)]

    return at


def test_a_wave(sim: Sim) -> None:
    events = starts(sim.run(3.0, wave(1.0, 2.0)))  # 2.6 s of waving, ten swings: one wave
    assert [e["gesture"] for e in events if e["kind"] == "motion"] == ["wave"]
    sim.run(1.0)
    assert "wave" in moves(sim.run(2.0, wave(1.0, 2.0)))  # rested: waving again is another


def test_a_small_tremble_is_no_wave(sim: Sim) -> None:
    assert "wave" not in moves(sim.run(2.0, wave(0.2, 3.0)))


def test_a_fist_shaking_is_no_wave(sim: Sim) -> None:
    assert "wave" not in moves(sim.run(2.0, wave(1.0, 2.0, gesture="Closed_Fist", fingers="00000")))


def test_a_wide_sweep_back_and_forth_is_swipes_not_a_wave(sim: Sim) -> None:
    got = moves(sim.run(3.0, wave(3.0, 0.8)))
    assert "wave" not in got and "swipe_left" in got


def circle(clockwise: bool, *, radius: float = 0.6, seconds: float = 1.0, size: float = 0.15):
    """A pointing hand at rest 0.4 s, then its index tip drawing circles of ``radius`` palm lengths."""

    def at(t: float):
        k = max(t - 0.4, 0.0)
        angle = 2 * math.pi * k / seconds * (1 if clockwise else -1)
        return [hand(0.66 + radius * size * math.cos(angle), 0.5 + radius * size * math.sin(angle), size=size,
                     gesture="Pointing_Up", fingers="01000")]

    return at


def test_circles_both_ways(sim: Sim) -> None:
    assert moves(sim.run(2.0, circle(True))) == ["circle_cw"]
    sim.run(1.5)
    assert moves(sim.run(2.0, circle(False))) == ["circle_ccw"]


def test_a_tiny_circle_is_no_circle(sim: Sim) -> None:
    assert not [n for n in moves(sim.run(2.0, circle(True, radius=0.1))) if n.startswith("circle")]


def test_a_circle_is_no_wave(sim: Sim) -> None:
    def open_circle(t: float):
        k = max(t - 0.4, 0.0)
        a = 2 * math.pi * k
        return [hand(0.66 + 0.09 * math.cos(a), 0.5 + 0.09 * math.sin(a), gesture="Open_Palm")]

    got = moves(sim.run(2.0, open_circle))
    assert "wave" not in got and "circle_cw" in got


def push(grow: float, seconds: float = 0.4):
    """An open palm at rest 0.4 s, then growing (towards the camera) by ``grow`` times over ``seconds``."""

    def at(t: float):
        k = min(max((t - 0.4) / seconds, 0.0), 1.0)
        return [hand(0.66, 0.5, size=0.15 * (1 + (grow - 1) * k), gesture="Open_Palm")]

    return at


def test_push_and_pull(sim: Sim) -> None:
    assert [n for n in moves(sim.run(1.5, push(1.5))) if n in ("push", "pull")] == ["push"]
    sim.run(1.5)
    assert [n for n in moves(sim.run(1.5, push(1 / 1.5))) if n in ("push", "pull")] == ["pull"]


def test_a_slow_lean_is_no_push(sim: Sim) -> None:
    assert "push" not in moves(sim.run(3.0, push(1.5, seconds=2.5)))


def test_a_fist_coming_closer_is_no_push(sim: Sim) -> None:
    def fist(t: float):
        k = min(max((t - 0.4) / 0.4, 0.0), 1.0)
        return [hand(0.66, 0.5, size=0.15 * (1 + 0.5 * k), gesture="Closed_Fist", fingers="00000")]

    assert "push" not in moves(sim.run(1.5, fist))


# ---------------------------------------------------------------------------------------------------- two hands
def two(d_from: float, d_to: float, seconds: float, size: float = 0.12, *, drop: tuple[float, float] = (0, 0)):
    """Two hands whose palm centres go from ``d_from`` to ``d_to`` palm lengths apart in ``seconds`` (after 0.4 s
    at rest); between the times in ``drop`` the right hand is not seen."""

    def at(t: float):
        k = min(max((t - 0.4) / seconds, 0.0), 1.0)
        half = (d_from + (d_to - d_from) * k) * size / 2
        left, right = hand(0.66 - half, 0.5, size=size, handedness="Left"), hand(0.66 + half, 0.5, size=size)
        return [left] if drop[0] <= t < drop[1] else [left, right]

    return at


def test_hands_apart(sim: Sim) -> None:
    events = [e for e in starts(sim.run(1.5, two(1.2, 4.0, 0.4))) if e["kind"] == "two_hands"]
    assert [e["gesture"] for e in events] == ["hands_apart"]
    assert events[0]["hand"] == "both"
    assert sim.engine.apart == pytest.approx(4.0, abs=0.2)  # how far apart they are, for the live view
    sim.run(0.5)
    assert sim.engine.apart is None  # no hands, no distance


def test_hands_together(sim: Sim) -> None:
    assert moves(sim.run(1.5, two(4.0, 1.2, 0.4))) == ["hands_together"]


def test_a_hand_dropping_out_for_a_moment_doesnt_spoil_it(sim: Sim) -> None:
    assert moves(sim.run(1.5, two(1.2, 4.0, 0.4, drop=(0.5, 0.7)))) == ["hands_apart"]


def test_two_hands_merging_as_they_meet_count_as_together(sim: Sim) -> None:
    def meet(t: float):
        k = min(max((t - 0.4) / 0.4, 0.0), 1.0)
        half = (4.0 - 1.8 * k) * 0.12 / 2  # they close to 2.2 palm lengths...
        left, right = hand(0.66 - half, 0.5, size=0.12, handedness="Left"), hand(0.66 + half, 0.5, size=0.12)
        return [left, right] if t < 0.8 else [left]  # ...and the recognizer sees one from then on

    assert moves(sim.run(2.0, meet)) == ["hands_together"]


def test_too_slow_is_not_a_two_hand_gesture(sim: Sim) -> None:
    assert moves(sim.run(3.5, two(1.2, 4.0, 2.5))) == []


def test_hands_moving_apart_are_not_swipes(sim: Sim) -> None:
    got = moves(sim.run(1.5, two(1.0, 7.0, 0.4)))
    assert "swipe_left" not in got and "swipe_right" not in got


def open_pair(rest: float, seconds: float, *, alone: float = 0.0):
    """Two open palms 1.2 palm lengths apart, at rest ``rest`` s, then spread to 4 palm lengths in ``seconds`` and held
    there; before ``alone`` s only the right hand is up."""

    def at(t: float):
        k = min(max((t - rest) / seconds, 0.0), 1.0)
        half = (1.2 + 2.8 * k) * 0.12 / 2
        left = hand(0.66 - half, 0.5, size=0.12, gesture="Open_Palm", handedness="Left")
        right = hand(0.66 + half, 0.5, size=0.12, gesture="Open_Palm")
        return [right] if t < alone else [left, right]

    return at


@pytest.mark.parametrize(("rest", "alone"), [
    (0.4, 0.0),  # together a moment, spread, then held apart
    (1.2, 0.0),  # together over a second before the spread
    (0.8, 0.5),  # one hand up first (its palm starts), the other joins it, then the spread
])
def test_two_open_hands_spreading_are_no_open_palm(sim: Sim, rest: float, alone: float) -> None:
    events = sim.run(rest + 2.5, open_pair(rest, 0.3, alone=alone))
    assert moves(events) == ["hands_apart"]
    assert "open_palm" not in names(events, "hold")  # never held on: a binding that waits for a hold never fires
    assert names(events) == (["open_palm", "hands_apart"] if alone else ["hands_apart"])


def test_a_shape_beside_a_resting_hand_still_counts(sim: Sim) -> None:
    resting = hand(0.2, 0.8, handedness="Left", fingers="00000")  # curled round a mouse: not open
    events = sim.run(2.0, [THUMB_UP, resting])
    assert names(events) == ["thumb_up"] and len(names(events, "hold")) >= 3


def test_an_open_palm_counts_again_once_the_two_hands_are_down(sim: Sim) -> None:
    sim.run(2.5, open_pair(0.4, 0.3))
    sim.run(1.2)  # both down, longer than two_hand_quiet
    assert names(sim.run(1.2, [PALM]), None)[:3] == ["open_palm", "open_palm", "open_palm"]


# ---------------------------------------------------------------------------------------------------- presence and view
def test_presence_changes_are_debounced(sim: Sim) -> None:
    events = sim.run(0.1, [PALM])  # 2 frames: not yet
    assert kinds(events, "presence") == []
    events = sim.run(0.2, [PALM])
    assert [e["hands"] for e in kinds(events, "presence")] == [1]
    events = sim.run(0.5, lambda t: [] if round(t * 20) == 3 else [PALM])  # a one-frame dropout
    assert kinds(events, "presence") == []
    other = hand(0.2, 0.5, handedness="Left")
    events = sim.run(0.5, [PALM, other])
    events += sim.run(0.5)
    assert [e["hands"] for e in kinds(events, "presence")] == [2, 0]


def test_the_view_says_what_each_hand_shows(sim: Sim) -> None:
    other = hand(0.2, 0.5, handedness="Left", fingers="01010")  # no shape at all
    sim.run(0.5, [VICTORY, other])
    first, second = sim.engine.view
    assert (first.hand, first.shape, first.score) == ("Right", "victory", 0.9) and 0.45 <= first.held <= 0.55
    assert (second.hand, second.shape, second.held) == ("Left", None, 0.0)
    assert first.track != second.track
    sim.run(0.1)
    assert sim.engine.view == []


# ---------------------------------------------------------------------------------------------------- control
def test_paused_reports_no_gestures(sim: Sim) -> None:
    sim.engine.pause(sim.t)
    events = sim.run(1.0, [VICTORY])
    assert kinds(events, "gesture") == [] and kinds(events, "presence")  # hands still counted
    sim.run(1.2)
    sim.engine.resume()
    assert names(sim.run(1.0, [VICTORY])) == ["victory"]


def test_a_timed_pause_ends_by_itself(sim: Sim) -> None:
    sim.engine.pause(sim.t, 2.0)
    assert sim.engine.snapshot(sim.t)["paused_for"] == 2.0
    assert names(sim.run(1.0, [FIST])) == []
    sim.run(1.5)
    assert names(sim.run(1.0, [FIST])) == ["closed_fist"]


def test_injected_gestures(sim: Sim) -> None:
    (event,) = sim.engine.inject("swipe_right")
    assert (event["gesture"], event["phase"], event["kind"]) == ("swipe_right", "start", "motion")
    (shape,) = sim.engine.inject("victory")
    assert shape["showing"] == shape["id"] > event["id"]


def test_snapshot_reports_state(sim: Sim) -> None:
    sim.run(0.5, [VICTORY])
    snap = sim.engine.snapshot(sim.t)
    assert snap["hands"] == 1 and snap["paused"] is False and snap["paused_for"] is None
    assert snap["last_gesture"]["gesture"] == "victory"


def test_frames_without_hands_are_cheap_and_quiet() -> None:
    engine = Engine(config.load(None))
    assert all(engine.update(Frame(t / 20)) == [] for t in range(100))


def test_rest_forgets_the_hands(sim) -> None:
    """The camera turned off: no hands in view any more."""
    sim.run(0.5, [VICTORY])
    events = sim.engine.rest()
    assert [(e["type"], e["hands"]) for e in events] == [("presence", 0)]
    assert sim.engine.snapshot(sim.t)["hands"] == 0 and sim.engine.view == []
    assert sim.engine.rest() == []  # nothing more to say


def test_hands_are_matched_to_the_nearest_track_even_with_stale_tracks_around(sim: Sim) -> None:
    left, right = hand(0.2, 0.5, handedness="Left"), hand(1.1, 0.5)
    sim.run(0.2, [left, right])
    sim.run(0.05, [hand(0.66, 0.2), hand(0.66, 0.8)])  # two hands far from both: two new tracks, the old ones linger
    assert len(sim.engine._tracks) == 4
    sim.run(0.05, [left, right])  # back where the first two were: matched to them again, no new tracks
    assert len(sim.engine._tracks) == 4
    sim.run(0.5, [left, right])
    assert sorted(tr.id for tr in sim.engine._tracks) == [1, 2]  # the stale ones timed out


# ---------------------------------------------------------------------------------------------------- the model
def test_mediapipe_results_become_hands() -> None:
    point = NS(x=0.5, y=0.25, z=-0.01)
    result = NS(hand_landmarks=[[point] * 21, [point] * 21],
                gestures=[[NS(category_name="Victory", score=0.83)], []],
                handedness=[[NS(category_name="Left", score=0.97)], [NS(category_name="Right", score=0.9)]])
    first, second = hands_from_result(result)
    assert (first.gesture, first.gesture_score, first.handedness) == ("Victory", 0.83, "Left")
    assert first.landmarks[0] == (0.5, 0.25, -0.01) and len(first.landmarks) == 21
    assert (second.gesture, second.gesture_score, second.handedness) == ("None", 0.0, "Right")
    assert hands_from_result(NS(hand_landmarks=[], gestures=[], handedness=[])) == []


class FakeRecognizer:
    """Returns scripted hands and counts its calls."""

    def __init__(self, script: list[int]) -> None:
        self.script, self.calls, self.closed = script, 0, False

    def __call__(self, rgb: object, ts_ms: int) -> list:
        count = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [hand(0.2 + 0.4 * i, 0.5) for i in range(count)]

    def close(self) -> None:
        self.closed = True


def test_the_scheduler_tracks_one_hand_cheaply_and_still_finds_a_second() -> None:
    wide = FakeRecognizer([0, 0, 1, 1, 1])
    narrow = FakeRecognizer([1])
    scheduler = Scheduler(wide, narrow, wide_every=3, sticky_ms=1500)
    modes = []
    for i in range(11):
        scheduler(None, i * 50)
        modes.append(scheduler.mode)
    # search, search, wide finds one -> two cheap narrow frames, then wide again, two narrow...
    assert modes[:3] == ["search", "search", "one"]
    assert (wide.calls, narrow.calls) == (5, 6)
    scheduler.close()
    assert wide.closed and narrow.closed


def test_after_two_hands_the_scheduler_looks_for_both_every_frame_a_while() -> None:
    wide = FakeRecognizer([2, 1])  # two hands, then one of them lost
    narrow = FakeRecognizer([1])
    scheduler = Scheduler(wide, narrow, wide_every=3, sticky_ms=500)
    for i in range(10):  # 50 ms apart: 450 ms, all within the half second after two were seen
        scheduler(None, i * 50)
    assert (wide.calls, narrow.calls) == (10, 0)
    for i in range(10, 16):  # then the one-hand recognizer takes over again
        scheduler(None, i * 50)
    assert narrow.calls > 0


def test_the_scheduler_goes_back_to_searching_when_the_hand_is_lost() -> None:
    wide = FakeRecognizer([1, 0])
    narrow = FakeRecognizer([0])
    scheduler = Scheduler(wide, narrow, wide_every=3, sticky_ms=1500)
    scheduler(None, 0)
    assert scheduler.mode == "one"
    assert scheduler(None, 50) == [] and scheduler.mode == "search"
    scheduler(None, 100)
    assert wide.calls == 2  # searching again with the two-hand recognizer


def test_a_single_recognizer_is_always_used_without_a_narrow_one() -> None:
    wide = FakeRecognizer([1])
    scheduler = Scheduler(wide, None, wide_every=3, sticky_ms=1500)
    for i in range(5):
        scheduler(None, i)
    assert wide.calls == 5


def test_the_engine_makes_a_frame_rate_independent_hold() -> None:
    clock = Clock()
    slow = Sim(Engine(config.load(None), wall=clock), clock, fps=10)
    events = kinds(slow.run(1.2, [VICTORY]), "gesture")
    assert [e["phase"] for e in events] == ["start", "hold", "hold"]  # 0.3, 0.6, 1.0 s at 10 frames a second


# ---------------------------------------------------------------------------------------------- found in review
def test_a_shape_started_late_skips_the_steps_it_already_passed(sim: Sim) -> None:
    sim.run(0.4, [THUMB_UP])  # its first showing
    sim.run(0.3)  # let go: shown again within the cooldown, its start waits until the cooldown is over
    events = kinds(sim.run(2.0, [THUMB_UP]), "gesture")
    start, first_hold = events[0], events[1]
    assert start["phase"] == "start" and start["held"] > 0.6  # started late, past the first step...
    assert first_hold["held"] >= 1.0 - 0.06  # ...so the next report is the next step, not a burst 50 ms later
    assert first_hold["ts"] - start["ts"] > 0.2


def test_the_last_hold_step_is_reported(sim: Sim, cfg: TrackerConfig) -> None:
    sim.engine.configure(tuned(cfg, max_hold=3.0))
    holds = [e["held"] for e in kinds(sim.run(4.0, [FIST]), "gesture") if e["phase"] == "hold"]
    assert holds[-1] >= 3.0 - 1e-6  # the 3 s step itself, and nothing after it
    assert len(holds) == len([h for h in holds if h <= 3.0 + 0.06])


def test_swipes_after_a_wave_are_swipes_again(sim: Sim) -> None:
    def two_swipes(t: float):
        if t < 0.4:
            x = 0.4
        elif t < 0.7:
            x = 0.4 + 0.5 * (t - 0.4) / 0.3
        elif t < 1.1:
            x = 0.9 - 0.5 * (t - 0.7) / 0.4
        elif t < 1.4:
            x = 0.4 + 0.5 * (t - 1.1) / 0.3
        else:
            x = 0.9
        return [hand(x, 0.5)]

    clock = Clock()
    fresh = Sim(Engine(sim.engine.config, wall=clock), clock)
    expected = moves(fresh.run(2.5, two_swipes))
    assert moves(sim.run(3.0, wave(1.0, 2.0))) == ["wave"]
    sim.run(3.0)  # rest: the wave is long over
    assert not sim.engine._lingering
    assert moves(sim.run(2.5, two_swipes)) == expected == ["swipe_right", "swipe_right"]


def test_a_longer_two_hand_gap_bridges_a_longer_dropout(sim: Sim, cfg: TrackerConfig) -> None:
    sim.engine.configure(tuned(cfg, two_hand_gap=0.6))
    assert moves(sim.run(1.6, two(1.2, 4.0, 0.5, drop=(0.45, 0.9)))) == ["hands_apart"]
