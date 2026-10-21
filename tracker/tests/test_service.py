"""The camera switch: the camera loop runs only while it is wanted."""

from handtracker.service import CameraSwitch


class FakeLoop:
    """A camera loop that runs until stopped (and keeps the camera a moment longer, like the real one)."""

    def __init__(self, made: list["FakeLoop"]) -> None:
        self.started = self.stopped = False
        self.alive = False
        self.failed = ""
        made.append(self)

    def start(self) -> None:
        self.started = self.alive = True

    def stop(self) -> None:
        self.stopped = True  # it lets go of the camera as it ends: `alive` turns false a little later

    def is_alive(self) -> bool:
        return self.alive


def make_switch(grace: float = 15.0) -> tuple[CameraSwitch, list[FakeLoop]]:
    made: list[FakeLoop] = []
    return CameraSwitch(lambda: FakeLoop(made), grace), made


def test_it_turns_on_when_wanted_and_only_once() -> None:
    switch, made = make_switch()
    assert switch.tick(False, 0.0) is None and not made
    assert switch.tick(True, 1.0) == "on" and switch.on and switch.since == 1.0
    assert switch.tick(True, 2.0) is None and len(made) == 1


def test_a_client_leaving_turns_it_off_after_the_grace() -> None:
    switch, made = make_switch(grace=15.0)
    switch.tick(True, 0.0)
    assert switch.tick(False, 10.0) is None and switch.on  # within the grace
    assert switch.tick(True, 20.0) is None and not made[0].stopped  # back in time: it never blinked
    assert switch.tick(False, 30.0) is None
    assert switch.tick(False, 44.0) is None and switch.on
    assert switch.tick(False, 45.0) == "off" and made[0].stopped and not switch.on


def test_a_pause_turns_it_off_at_once() -> None:
    switch, made = make_switch()
    switch.tick(True, 0.0)
    assert switch.tick(False, 1.0, at_once=True) == "off" and made[0].stopped


def test_it_waits_for_the_last_loop_to_let_go_of_the_camera() -> None:
    switch, made = make_switch(grace=0.0)
    switch.tick(True, 0.0)
    assert switch.tick(False, 1.0) == "off"
    assert switch.tick(True, 2.0) is None and len(made) == 1  # the first still holds the camera
    made[0].alive = False
    assert switch.tick(True, 3.0) == "on" and len(made) == 2


def test_a_camera_that_stays_unusable_is_reported_and_not_retried() -> None:
    switch, made = make_switch()
    switch.tick(True, 0.0)
    made[0].alive, made[0].failed = False, "no frames from /dev/video0 for 60 s"
    assert switch.tick(True, 1.0) is None
    assert switch.failed == "no frames from /dev/video0 for 60 s" and len(made) == 1 and not switch.on


def test_the_live_view_is_made_only_while_watched_and_at_its_own_pace() -> None:
    from handtracker.gestures import HandView
    from handtracker.service import View
    from tests.synth import hand

    view = View(fps=15, picture_fps=5)
    assert not view.wanted(0.0) and not view.watched  # nobody watches: nothing is made
    view.landmarks = 1
    assert view.wanted(0.0) and view.watched and not view.picture_wanted(0.0)
    shown = [HandView(track=3, hand="Left", shape="victory", score=0.9, held=0.45)]
    frame = view.frame(wall=1.5, size=(640, 480), hands=[hand(0.5, 0.5)], shown=shown, fps=19.5, mode="one")
    assert (frame["seq"], frame["w"], frame["h"], frame["mode"]) == (1, 640, 480, "one")
    (h,) = frame["hands"]
    assert (h["track"], h["hand"], h["shape"], h["held"]) == (3, "Left", "victory", 0.45)
    assert len(h["pts"]) == 21 and all(len(p) == 2 for p in h["pts"]) and "jpg" not in frame
    assert frame["apart"] is None  # one hand: no distance between two
    assert view.frame(wall=1.7, size=(640, 480), hands=[], shown=[], fps=19.5, mode="two",
                      apart=2.4567)["apart"] == 2.46
    assert not view.wanted(0.5 / 15) and view.wanted(1 / 15)  # at most live_view.fps a second
    view.images = 1
    assert view.picture_wanted(0.1)
    frame = view.frame(wall=1.6, size=(640, 480), hands=[], shown=[], fps=19.5, mode="search", jpeg=b"\xff\xd8")
    assert frame["jpg"] == "/9g=" and frame["hands"] == []
    assert not view.picture_wanted(0.1 + 0.5 / 5)


def test_the_live_views_pace_keeps_up_with_uneven_frames() -> None:
    from handtracker.service import Pace

    pace = Pace(20)  # 20 a second, frames 33 and 66 ms apart (20 of the camera's 30)
    times = [i * 0.1 + d for i in range(20) for d in (0.0, 1 / 30)]
    assert sum(pace.take(t) for t in times) >= 38  # all but the first ones' wait: none dropped for being early
    slow = Pace(5)
    assert sum(slow.take(i / 20) for i in range(40)) <= 11  # two seconds at 20 fps: 5 a second (and a burst of 2)
