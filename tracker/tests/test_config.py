"""The settings: defaults.yaml as shipped, the Pi's tracker.yaml laid over it, every wrong key or value refused by
name, what reloads live; and the vocabulary and the version."""

from pathlib import Path

import pytest

from handtracker import build_of, config
from handtracker.vocabulary import GESTURES, SHAPES, describe


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tracker.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_shipped_defaults_are_complete() -> None:
    cfg = config.load(None)
    assert cfg.server.port == 8770 and cfg.camera.mirror and cfg.recognition.static_hold == 0.3
    assert cfg.ignore == frozenset() and cfg.camera.controls == {"exposure_dynamic_framerate": 0}


def test_the_pis_file_holds_only_what_changed(tmp_path: Path) -> None:
    cfg = config.load(write(tmp_path, "camera:\n  mirror: false\n  controls: {brightness: 4}\nignore: [pinch]\n"))
    assert not cfg.camera.mirror and cfg.camera.fps == 30  # the rest of the section stays
    assert cfg.camera.controls == {"exposure_dynamic_framerate": 0, "brightness": 4}  # mappings merge key by key
    assert cfg.ignore == {"pinch"}
    assert config.load(write(tmp_path, "")) == config.load(None)  # an empty file: the defaults


@pytest.mark.parametrize(("text", "said"), [
    ("camera:\n  mirorr: false\n", "camera.mirorr: Extra inputs are not permitted"),
    ("cameras: {}\n", "cameras: Extra inputs are not permitted"),
    ("processing:\n  fps: 0\n", "processing.fps: Input should be greater than or equal to 1"),
    ("server:\n  port: 70000\n", "server.port: Input should be less than or equal to 65535"),
    ("recognition:\n  pinch_on: 0.5\n  pinch_off: 0.4\n", "pinch_off must be at least pinch_on"),
    ("ignore: [jazz_hands]\n", "unknown gesture(s) jazz_hands"),
    ("logging:\n  level: LOUD\n", "logging.level"),
    ("- a list\n", "must hold a mapping of sections"),
    ("camera: [\n", "is not valid YAML"),
])
def test_a_wrong_key_or_value_is_refused_by_name(tmp_path: Path, text: str, said: str) -> None:
    with pytest.raises(config.ConfigError, match=None) as caught:
        config.load(write(tmp_path, text))
    assert said in str(caught.value)


def test_every_problem_is_said_at_once(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError) as caught:
        config.load(write(tmp_path, "camera:\n  fps: 0\nprocessing:\n  fps: 0\n"))
    assert "camera.fps" in str(caught.value) and "processing.fps" in str(caught.value)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        config.load(tmp_path / "missing.yaml")


def test_only_recognition_and_ignore_apply_without_a_restart(tmp_path: Path) -> None:
    before = config.load(None)
    after = config.load(write(tmp_path, "recognition:\n  static_hold: 0.5\nignore: [pinch]\ncamera:\n  fps: 15\n"))
    assert after.restart_needed(before) == ["camera"]


def test_the_watcher_notices_a_change(tmp_path: Path) -> None:
    path = write(tmp_path, "")
    watcher = config.Watcher(path)
    assert not watcher.changed()
    path.write_text("ignore: [pinch]\n", encoding="utf-8")
    assert watcher.changed() and not watcher.changed()


def test_the_build_in_the_version() -> None:
    assert build_of("") == ""
    assert build_of(" 2d723c272a71f0e5b1c9d8a7e6f5a4b3c2d1e0f9\n") == "2d723c272a71"  # CI: the whole commit
    assert build_of("f29fc0012345-dirty") == "f29fc0012345-dirty"  # push-pi.sh from a checkout with changes
    assert build_of("local") == "local"


def test_the_vocabulary() -> None:
    assert {"listen", "stop"}.isdisjoint(GESTURES)
    assert {"ok_sign", "pinch", "call_me", "three", "victory"} <= SHAPES
    entries = describe()
    assert [e["name"] for e in entries] == list(GESTURES)
    assert all(e["title"] and e["how"].endswith(".") and e["kind"] in ("shape", "motion", "two_hands")
               for e in entries)
