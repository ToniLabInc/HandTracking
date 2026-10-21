"""Shared fixtures: the shipped defaults (defaults.yaml) and an engine driven by a fake clock."""

import pytest

from handtracker import config
from handtracker.config import TrackerConfig
from handtracker.gestures import Engine
from tests.synth import Clock, Sim


@pytest.fixture
def cfg() -> TrackerConfig:
    """The settings as shipped (defaults.yaml alone)."""
    return config.load(None)


@pytest.fixture
def sim(cfg: TrackerConfig) -> Sim:
    """An engine with the default settings, fed at 20 fps."""
    clock = Clock()
    return Sim(Engine(cfg, wall=clock), clock)
