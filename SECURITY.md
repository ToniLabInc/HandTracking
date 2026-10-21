# Security policy

## Reporting a vulnerability

Please report it privately: on this repository's **Security** tab choose **Report a vulnerability** (GitHub shows the
report to the maintainers alone). Don't open a public issue or pull request for it. Say what you found, the tracker's
version (`curl http://<pi>:8770/health` prints it) and how to reproduce it. You'll get an answer as soon as a
maintainer can, and credit in the fix unless you'd rather not have it.

Only the latest release and `main` are supported.

## What the tracker defends

The tracker is a camera on your network, so it promises little, and keeps each promise in code you can read
(`tracker/handtracker/server.py` for the guard, `deploy/compose.yaml` for the container):

- **Video stays on the Pi, with one exception you switch on.** Orion gets gestures, and your hands' landmarks while its
  HUD shows the live view. The camera's picture (a 320-pixel-wide JPEG, up to 10 a second) goes only to a client that
  asks for `orion://hands/camera`, which Orion's HUD does only when you turn the picture on in its HANDS view. The
  tracker never writes frames to disk, and Orion hands the latest one to the HUD without keeping it.
- **The camera is on only while someone follows the tracker** and gestures aren't paused; otherwise it is closed and the
  model unloaded.
- **Every request but `/health` needs the bearer token** (`deploy/secrets/hands_token`, mode 600), and the endpoint
  answers private, loopback and link-local addresses only (`server.lan_only`), whatever the `Host` header says.
- **The container is boxed in**: it runs as your user, with a read-only filesystem, no capabilities and
  `no-new-privileges`; it has only the camera device.

Not defended against: anyone who has the token, someone on your network who can read it off a plain-HTTP connection
(use a network you trust, or put the endpoint behind your own TLS proxy), and a Pi someone else has root on.
Don't expose port 8770 to the internet.

## Handling the token

The token is a secret like a password. `install-pi.sh` creates it; keep it out of git (the repository's `.gitignore`
already does) and out of screenshots and issues. If it leaks, replace the content of `deploy/secrets/hands_token`
with a new random string of at least 16 characters, run `docker compose restart tracker` on the Pi, and set the new
token in Orion.
