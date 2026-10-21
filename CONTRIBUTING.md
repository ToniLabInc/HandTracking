# Contributing to ORION hands

Thanks for helping. This is a small project: a tracker that runs on a Raspberry Pi, and the deploy scripts around it.
Bug reports, new gestures, better thresholds and fixes for other cameras and boards are all welcome. By taking part you
agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). For a security problem, don't open an issue: see
[SECURITY.md](SECURITY.md).

## Before you start

For anything bigger than a fix, open an issue first so we can agree on the shape of it. Gestures and their thresholds
are easy to get subtly wrong, and a short discussion saves a rewrite.

## Setting up

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). No camera, model or OpenCV is needed to work on the
logic: the tests drive the gesture engine with synthetic hands.

```sh
cd tracker
uv run --no-project --with-requirements requirements-test.txt python -m pytest   # the tests
uvx ruff check .                                                                  # lint (120 columns)
shellcheck ../deploy/install-pi.sh ../deploy/push-pi.sh                           # the shell scripts
docker build -t orion-hands:latest .                                              # the image, if you changed it
```

CI runs the same checks, builds the image for arm64 and amd64 and smoke-tests it, so a pull request that passes the
commands above should pass CI.

To try the recognizer on photos without a camera: `python -m handtracker recognize photo.jpg` (inside the image, or in
an environment with the packages of `requirements.txt`). `python -m handtracker bench` times each stage on live frames.

## How the code is laid out

| | |
|---|---|
| `tracker/handtracker/vocabulary.py` | the 22 gestures: their names (shared with Orion), titles and how to do them |
| `tracker/handtracker/gestures.py` | the engine: turns hand landmarks into gestures (holds, swipes, circles, two hands) |
| `tracker/handtracker/vision.py`, `service.py` | the camera, MediaPipe, and the on-demand camera loop |
| `tracker/handtracker/server.py` | the MCP endpoint, its token and LAN guard, and the live view |
| `tracker/handtracker/defaults.yaml`, `config.py` | every setting, with its default, and how a Pi's file is laid over it |
| `tracker/tests/` | the tests; `synth.py` makes synthetic hands, `test_server.py` serves MCP to a real client |
| `deploy/` | what runs on the Pi (Docker Compose, `install-pi.sh`) and what deploys to it from a PC (`push-pi.sh`) |

## Changing things

- **A gesture's thresholds or timings** live in `defaults.yaml`, with a comment each. Change a number there, not in the
  code, and add or adjust a test in `tests/test_gestures.py` that fails without your change.
- **A new gesture** needs an entry in `vocabulary.py`, its recognition in `gestures.py`, tests with synthetic hands, a
  row in the README's table, and the same name in Orion (the gesture's name is the contract between the two projects:
  [Orion](https://github.com/ToniLabInc/Orion) learns the vocabulary from the tracker's `orion://gestures`, so a new
  name appears there without a change, but its HUD and defaults should know about it).
- **The MCP surface** (tools and `orion://` resources) is what Orion and other clients depend on. Add to it freely;
  don't rename or remove anything without saying so in the pull request.
- **Never commit camera frames or recordings**, or anything from a real Pi: `deploy/config/`, `deploy/secrets/`,
  `deploy/compose.override.yaml` and `deploy/pi.yaml` are ignored on purpose. Use made-up addresses and tokens in tests.
- **Dependencies** are pinned with hashes in `tracker/requirements.txt`, listed in `requirements.in`. To change one,
  edit `requirements.in` and run the command at the top of `requirements.txt`; keep it to licences compatible with MIT
  (Apache 2.0, MIT, BSD).

## Pull requests

- Keep each pull request to one change, with a description of what it does and why, and how you tried it (on a Pi? with
  which camera?).
- Match the surrounding code: type hints, a docstring where one helps, comments that say why, lines up to 120 columns.
- Add or update tests, and the README when behaviour a user sees changes. Add a line to [CHANGELOG.md](CHANGELOG.md)
  under "Unreleased".
- Write commit messages that say what changed for the reader of the history, in the imperative or as a plain
  description ("Spreading both hands is no open palm"), not "fix stuff".

## Licence of your contribution

The project is MIT ([LICENSE](LICENSE)). By opening a pull request you agree that your contribution is released under
the same licence.
