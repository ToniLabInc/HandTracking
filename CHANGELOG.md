# Changelog

All notable changes to ORION hands. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the version is the tracker's (`tracker/pyproject.toml`, reported by `/health`). Dates are those of the commits.

## Unreleased

## 3.0.0 - 2026-10-08

### Changed
- The tracker is the MCP server itself, and every setting is in YAML (`defaults.yaml`, with a Pi's own
  `config/tracker.yaml` laid over it).
- A hand shape counts after it has been held still for `recognition.static_hold`, and the vocabulary says so.
- While two open hands are up (a two-hand gesture), no hand shape counts: spreading both hands is no open palm.

### Added
- The live view says how far apart two hands are, and the vocabulary the distances that count.
- CI smoke-tests the image before pushing it, against the gesture vocabulary.

### Removed
- The separate bridge and the updater: the tracker serves Orion directly.

## 2.0.0 - 2026-10-07

### Added
- 22 gestures that Orion maps, a live view of your hands, and deploys over SSH (`deploy/push-pi.sh`).

## 1.0.0 - 2026-10-05

### Added
- Gestures from the Raspberry Pi's camera, as intents for Orion over MCP.
