# ORION hands

[![CI](https://github.com/ToniLabInc/HandTracking/actions/workflows/ci.yml/badge.svg)](https://github.com/ToniLabInc/HandTracking/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Hand-gesture control for [ORION](https://github.com/ToniLabInc/Orion). A Raspberry Pi 5 with a USB camera
recognises 22 gestures with Google's MediaPipe and serves them as an **MCP server** on your network; Orion follows it
and does what you chose for each gesture (Settings → Abilities → hands): play/pause, the volume, a panel, a skill, or
any tool of any MCP server.

Orion never gets video, only gestures: while his HUD shows the live view of your hands it also gets their landmarks, and
the camera's picture only if you turn it on in that view. Nothing is recorded. **The camera is on only while Orion
follows the tracker** and gestures aren't paused; otherwise it's closed, the model unloaded, and the Pi idle. While it
watches, the tracker uses at most about a quarter of the Pi's CPU (one of its four cores).

## Gestures

| Gesture | How | Orion's default |
|---|---|---|
| `open_palm` | a flat hand, fingers up | HANDS (the live view) on / off, held 1 s |
| `closed_fist` | a fist | sound off / on, held 1 s |
| `pointing_up` | only the index finger up | COMMS on / off |
| `victory` | index and middle fingers in a V | play / pause |
| `three` | three fingers up | VITALS on / off |
| `thumb_up` / `thumb_down` | a thumbs-up / -down | volume up / down, repeating while held |
| `i_love_you` | thumb, index and little finger out | the Neural Map on / off |
| `call_me` | thumb and little finger out | microphone off / on, held 0.6 s |
| `ok_sign` | thumb and index in a ring | a new conversation, held 0.6 s |
| `pinch` | thumb and index tips touching | free |
| `swipe_left` / `swipe_right` | sweep a hand sideways, quickly | previous / next track |
| `swipe_up` / `swipe_down` | the same, up or down | free |
| `wave` | an open hand swinging side to side | the `morning-briefing` skill |
| `circle_cw` / `circle_ccw` | draw a circle with the index finger | free |
| `push` / `pull` | the open palm towards / away from the camera | dismiss the newest panel / free |
| `hands_apart` | both hands together, then spread apart | every panel |
| `hands_together` | both hands apart, then brought together | clear the screen |

Directions are yours (the picture is mirrored first). A hand shape counts once held still for 0.3 s and is reported
again while held (at 0.6, 1, 1.5, 2 s…), which is how a binding can wait for a longer hold or repeat. A moving hand
never holds a shape, and while two open hands are up (a two-hand gesture) no hand shape counts, so spreading your
hands is never taken for an open palm. Every timing and distance is in
[`tracker/handtracker/defaults.yaml`](tracker/handtracker/defaults.yaml).

## Setup on the Pi

Raspberry Pi OS (64-bit) with a USB camera:

```sh
git clone https://github.com/ToniLabInc/HandTracking.git && cd HandTracking/deploy
./install-pi.sh                   # Docker if missing, this Pi's own files, the image, then it starts the tracker
cat secrets/hands_token           # the token Orion needs
```

`install-pi.sh` keeps this Pi's own files out of git: `config/tracker.yaml` (your settings, only what you change from
`defaults.yaml`), `secrets/hands_token` (mode 600) and `compose.override.yaml` (the camera's device path). The
container runs as your user with only the camera, a read-only filesystem and no capabilities, restarts by itself, and
reports its health at `/health`. Logs: `docker compose logs -f tracker` (one line per gesture).

Settings: in `deploy/config/tracker.yaml`, under the same sections as `defaults.yaml`:

```yaml
camera:
  mirror: false          # left and right look swapped
recognition:
  static_hold: 0.4       # hold a shape longer before it counts
ignore: [pinch, swipe_up]
```

`recognition` and `ignore` apply when you save; anything else after `docker compose restart tracker`.

## Deploying from the PC

Copy [`deploy/pi.example.yaml`](deploy/pi.example.yaml) to `deploy/pi.yaml` (git ignores it) and name your Pi in it,
then:

```sh
deploy/push-pi.sh              # copies tracker/ and deploy/ to the Pi, builds there and restarts the tracker
deploy/push-pi.sh --rollback   # back to the image that ran before
```

## Adding it to Orion

In Orion's `data/config.yaml` (applied within a second):

```yaml
mcp_servers:
  hands:
    url: http://raspberrypi.local:8770/mcp     # the Pi's name on your network, or its address
    headers: {Authorization: "Bearer <the Pi's deploy/secrets/hands_token>"}
    intents: true                              # lets the gestures reach Orion; only you can set it
```

Then Settings → Abilities → hands shows the switch, a pause, and every gesture with what it does. HANDS on Orion's
stage ("Orion, show me my hands") shows your hands as the camera sees them.

What the server offers, for other MCP clients: the tools `hands_status`, `hands_gestures`, `hands_pause {minutes?}` and
`hands_resume`; the resources `orion://intents` (the latest gestures; a `subscriptions/listen` stream on it is what
keeps the camera on), `orion://gestures` (the vocabulary), and `orion://hands` / `orion://hands/camera` (the live view).
It answers only LAN addresses, and every request but `/health` needs the bearer token.

## Development

```sh
cd tracker
uv run --no-project --with-requirements requirements-test.txt python -m pytest   # no camera, model or OpenCV needed
uvx ruff check .
docker build -t orion-hands:latest .                                            # the image
```

The tests drive the gesture engine with synthetic hands (`tests/synth.py`) and serve the MCP endpoint to a real MCP
client. `python -m handtracker recognize photo.jpg` runs the recognizer on photos; `python -m handtracker bench` times
each stage on live frames. CI lints, tests, builds the image for arm64 and amd64 and pushes it to GHCR; the Pi builds
its own.

## Contributing and security

Bug reports and pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md), and the
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security problem, privately, see [SECURITY.md](SECURITY.md) (it also
says what the tracker promises about your camera). Changes are in [CHANGELOG.md](CHANGELOG.md).

## Licence

ORION hands is MIT ([LICENSE](LICENSE)). It builds on MediaPipe and its gesture model (Apache 2.0); OpenCV
(headless), Apache 2.0, with FFmpeg (LGPL 2.1); and the MCP Python SDK, numpy, PyYAML, pydantic, Starlette and uvicorn
(MIT or BSD). Each keeps its own licence.
