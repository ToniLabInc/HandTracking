"""The tracker's settings: defaults.yaml (shipped with the code) holds every one, its default and what it means; the
Pi's /config/tracker.yaml holds only what was changed there, laid over it key by key. Code holds the types and ranges
of the settings, never a default.

``load`` checks everything at once: a wrong key or value stops the tracker (``ConfigError``, every problem named by its
key). ``recognition`` and ``ignore`` are also reloaded while it runs (``Watcher``); a broken edit is then logged and the
last good values stay. The rest needs a restart.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from handtracker.vocabulary import GESTURES

DEFAULTS: Final = Path(__file__).with_name("defaults.yaml")
LIVE: Final = frozenset({"recognition", "ignore"})  # the parts that apply without a restart

type Seconds = Annotated[float, Field(ge=0, le=3600)]
type Positive = Annotated[float, Field(ge=0)]
type Share = Annotated[float, Field(ge=0, le=1)]
type Rate = Annotated[float, Field(ge=0, le=60)]


class ConfigError(ValueError):
    """The settings can't be used; the message names every problem, one per line."""


class Section(BaseModel):
    """A part of the settings: every key is needed (defaults.yaml has them all), and an unknown one is refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Server(Section):
    host: Annotated[str, Field(min_length=1)]
    port: Annotated[int, Field(ge=1, le=65535)]
    path: Annotated[str, Field(pattern=r"^/\S*$")]
    token_file: Path
    lan_only: bool
    max_listeners: Annotated[int, Field(ge=1, le=256)]
    dead_client_after_s: Annotated[float, Field(ge=5, le=3600)]


class Camera(Section):
    device: Annotated[str, Field(min_length=1)]
    width: Annotated[int, Field(ge=160, le=3840)]
    height: Annotated[int, Field(ge=120, le=2160)]
    fps: Annotated[int, Field(ge=1, le=120)]
    fourcc: Annotated[str, Field(min_length=4, max_length=4)]
    controls: dict[str, int]
    buffer_frames: Annotated[int, Field(ge=1, le=10)]
    mirror: bool
    idle_grace_s: Seconds
    always_on: bool
    reopen_every_s: Annotated[float, Field(gt=0, le=60)]
    exit_after_s: Annotated[float, Field(ge=5, le=3600)]
    fresh_s: Annotated[float, Field(gt=0, le=60)]
    warmup_s: Seconds


class Processing(Section):
    fps: Annotated[float, Field(ge=1, le=60)]
    idle_fps: Rate
    idle_after_s: Seconds
    deep_idle_fps: Rate
    deep_idle_after_s: Annotated[float, Field(ge=1, le=3600)]
    full_rate_while_watched: bool
    infer_width: Annotated[int, Field(ge=0, le=3840)]


class Model(Section):
    path: Path
    num_hands: Annotated[int, Field(ge=1, le=2)]
    min_detection: Share
    min_presence: Share
    min_tracking: Share
    wide_every: Annotated[int, Field(ge=2, le=30)]
    two_hand_sticky_s: Annotated[float, Field(ge=0, le=10)]


class LiveView(Section):
    fps: Annotated[float, Field(gt=0, le=60)]
    picture_fps: Annotated[float, Field(gt=0, le=60)]
    picture_width: Annotated[int, Field(ge=80, le=1920)]
    picture_quality: Annotated[int, Field(ge=10, le=95)]
    lease_s: Annotated[float, Field(gt=0, le=600)]


class Intents(Section):
    keep: Annotated[int, Field(ge=1, le=500)]


class Status(Section):
    every_s: Annotated[float, Field(ge=1, le=600)]
    log_every_s: Annotated[float, Field(ge=1, le=86400)]


class Logging(Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Debug(Section):
    inject_tool: bool


class Recognition(Section):
    """How gestures are recognised: times in seconds, sizes and distances in palm lengths (see defaults.yaml)."""

    min_score: Share
    static_hold: Annotated[float, Field(ge=0, le=5)]
    hold_ratio: Annotated[float, Field(gt=0, le=1)]
    max_gap: Seconds
    still_speed: Positive
    cooldown: Seconds
    repeat_every: Annotated[float, Field(ge=0.2, le=60)]
    max_hold: Annotated[float, Field(ge=0, le=30)]
    pinch_on: Positive
    pinch_off: Positive
    ok_touch: Positive
    settle: Seconds
    swipe_min_time: Annotated[float, Field(gt=0, le=3)]
    swipe_max_time: Annotated[float, Field(gt=0, le=3)]
    swipe_distance: Positive
    swipe_min_travel: Share
    swipe_straightness: Positive
    swipe_cooldown: Seconds
    swipe_return_block: Seconds
    wave_window: Seconds
    wave_swings: Annotated[int, Field(ge=1, le=10)]
    wave_min_swing: Positive
    wave_max_swing: Positive
    circle_window: Seconds
    circle_turn: Annotated[float, Field(ge=0, le=720)]
    circle_min_radius: Positive
    push_window: Seconds
    push_scale: Annotated[float, Field(gt=1)]
    motion_cooldown: Seconds
    two_hand_window: Seconds
    together_distance: Positive
    apart_distance: Positive
    two_hand_gap: Seconds
    two_hand_cooldown: Seconds
    two_hand_quiet: Seconds
    presence_debounce: Seconds

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Values that must agree with each other."""
        problems = [message for wrong, message in (
            (self.pinch_off < self.pinch_on, "pinch_off must be at least pinch_on"),
            (self.swipe_min_time >= self.swipe_max_time, "swipe_min_time must be less than swipe_max_time"),
            (self.apart_distance <= self.together_distance, "apart_distance must be more than together_distance"),
            (self.wave_max_swing <= self.wave_min_swing, "wave_max_swing must be more than wave_min_swing"),
        ) if wrong]
        if problems:
            raise ValueError("; ".join(problems))
        return self


class TrackerConfig(Section):
    """Every setting of the tracker (the gesture engine reads ``recognition`` and ``ignore``)."""

    server: Server
    camera: Camera
    processing: Processing
    model: Model
    live_view: LiveView
    intents: Intents
    status: Status
    logging: Logging
    debug: Debug
    recognition: Recognition
    ignore: frozenset[str]

    @field_validator("ignore")
    @classmethod
    def _known(cls, value: frozenset[str]) -> frozenset[str]:
        """Only gestures the tracker knows can be ignored."""
        if unknown := sorted(value - set(GESTURES)):
            raise ValueError(f"unknown gesture(s) {', '.join(unknown)}; the gestures are: {', '.join(GESTURES)}")
        return value

    def restart_needed(self, other: "TrackerConfig") -> list[str]:
        """The parts that differ from ``other`` and apply only after a restart."""
        return [name for name in type(self).model_fields if name not in LIVE and getattr(self, name) !=
                getattr(other, name)]


def merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """``over`` laid over ``base``: mappings merge key by key, anything else replaces what was there."""
    out = dict(base)
    for key, value in over.items():
        out[key] = merge(out[key], value) if isinstance(value, Mapping) and isinstance(out.get(key), Mapping) else value
    return out


def _read(path: Path) -> dict[str, Any]:
    """A YAML file's mapping (an empty file is an empty mapping).

    Raises:
        ConfigError: it isn't YAML, or not a mapping.
        OSError: it can't be read.
    """
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from None
    if doc is None:
        return {}
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{path} must hold a mapping of sections (server:, camera:, ... as in defaults.yaml)")
    return dict(doc)


def load(path: Path | None) -> TrackerConfig:
    """The settings: defaults.yaml, with ``path`` (the Pi's own, None for the defaults alone) laid over it.

    Raises:
        ConfigError: a key or value is wrong (every problem, one per line).
        OSError: a file can't be read (a missing tracker.yaml too: install-pi.sh makes it).
    """
    doc = _read(DEFAULTS) if path is None else merge(_read(DEFAULTS), _read(path))
    try:
        return TrackerConfig.model_validate(doc)
    except ValidationError as e:
        where = path or DEFAULTS
        raise ConfigError("\n".join(f"{where}: {'.'.join(map(str, err['loc'])) or '(top)'}: {err['msg']}"
                                    for err in e.errors())) from None


class Watcher:
    """Notices when a settings file changed on disk (one ``stat`` per check)."""

    def __init__(self, path: Path) -> None:
        """Watch ``path``; the version there now counts as seen."""
        self.path = path
        self._seen = self._signature()

    def _signature(self) -> tuple[int, int, int] | None:
        """The file's identity and version: inode, size, modification time (None when missing)."""
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_ino, st.st_size, st.st_mtime_ns

    def changed(self) -> bool:
        """Whether the file changed since the last call."""
        now = self._signature()
        if now == self._seen:
            return False
        self._seen = now
        return now is not None
