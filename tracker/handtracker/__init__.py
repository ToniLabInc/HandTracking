"""ORION hand-gesture tracker: a camera, MediaPipe's gesture recognizer, and gestures for Orion as an MCP server."""

import os

__version__ = "3.0.0"


def build_of(raw: str) -> str:
    """The build HANDS_BUILD names: its commit cut to 12 characters, and a mark after it kept (push-pi.sh's -dirty)."""
    commit, dash, mark = raw.strip().partition("-")
    return commit[:12] + dash + mark


BUILD = build_of(os.environ.get("HANDS_BUILD", ""))  # the commit the image was built from (set by CI or push-pi.sh)
VERSION = f"{__version__}+{BUILD}" if BUILD else __version__
