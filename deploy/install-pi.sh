#!/usr/bin/env bash
# Set up or update ORION hands on a Raspberry Pi (Raspberry Pi OS / Debian, arm64): Docker, this Pi's own files, the
# image (built here, from ../tracker), and the tracker.
#
# Run it from this folder, as your normal user (it uses sudo where needed):  ./install-pi.sh
# From the PC, deploy/push-pi.sh copies a new version over and runs it. Safe to run again: config/tracker.yaml,
# secrets/hands_token and compose.override.yaml are made once and then left as they are.
# BUILD=<label> names the build in the tracker's version (push-pi.sh passes the commit).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
IMAGE=orion-hands:latest

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

# 1. Docker (the official convenience script), and the groups for this user ---------------------------------
if ! command -v docker >/dev/null 2>&1; then
    say "installing Docker (get.docker.com)"
    curl -fsSL https://get.docker.com | sudo sh
fi
if ! systemctl is-active --quiet docker; then
    sudo systemctl enable --now docker >/dev/null
fi
for group in docker video; do
    if ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$group"; then
        say "adding $USER to the $group group"
        sudo usermod -aG "$group" "$USER"
    fi
done

# Docker commands as this user, even before a new login picks up the docker group.
dk() {
    if docker info >/dev/null 2>&1; then
        docker "$@"
    else
        sg docker -c "$(printf '%q ' docker "$@")"
    fi
}

# 2. This Pi's own files ---------------------------------------------------------------------------------------
mkdir -p config secrets
if [[ ! -f secrets/hands_token ]]; then
    umask 077
    if [[ -f .env ]] && grep -q '^HANDS_TOKEN=.' .env; then  # version 2 kept it in .env: Orion keeps working
        sed -n 's/^HANDS_TOKEN=//p' .env > secrets/hands_token
        say "secrets/hands_token: the token version 2 kept in .env"
    else
        python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > secrets/hands_token
        say "secrets/hands_token: a new token (Orion sends it: see the README, \"Adding it to Orion\")"
    fi
    umask 022
fi
if [[ ! -f config/tracker.yaml ]]; then
    say "config/tracker.yaml: your settings (none changed yet)"
    cat > config/tracker.yaml <<'EOF'
# ORION hands on this Pi: only the settings you change, under their sections, as in the tracker's defaults.yaml
# (tracker/handtracker/defaults.yaml, every setting with its default and what it means). For example:
#
#   camera:
#     mirror: false          # left and right look swapped
#   ignore: [pinch]          # gestures never reported
#
# recognition and ignore apply when this file is saved; anything else after: docker compose restart tracker
EOF
fi
if [[ ! -f compose.override.yaml ]]; then
    camera=$(find /dev/v4l/by-id -name '*-video-index0' 2>/dev/null | sort | head -n 1)
    if [[ -z "$camera" ]]; then
        echo "no camera under /dev/v4l/by-id: plug the USB camera in and run this again" >&2
        exit 1
    fi
    say "compose.override.yaml: the camera $camera, user $(id -un)"
    cat > compose.override.yaml <<EOF
# This Pi's own (written by install-pi.sh): its camera, and the user the tracker runs as (who owns config/ and secrets/)
services:
  tracker:
    devices: ["$camera:/dev/video0"]
    group_add: ["$(getent group video | cut -d: -f3)"]
    user: "$(id -u):$(id -g)"
EOF
fi

# 3. The image, built here (the one running now is kept as orion-hands:previous, for push-pi.sh --rollback) ------
if dk image inspect "$IMAGE" >/dev/null 2>&1; then
    dk tag "$IMAGE" orion-hands:previous
fi
say "building $IMAGE (a few minutes the first time, much less with its cache)"
dk build --build-arg BUILD="${BUILD:-}" -t "$IMAGE" ../tracker

# 4. Start it, and wait until it is healthy --------------------------------------------------------------------
say "starting the tracker"
if ! dk compose up -d --remove-orphans --wait; then
    dk compose logs --tail 40 tracker >&2
    echo "the tracker didn't come up healthy (its log is above)" >&2
    exit 1
fi
dk image prune -f >/dev/null
port=$(dk compose exec -T tracker python -c \
    "from pathlib import Path; from handtracker import config; print(config.load(Path('/config/tracker.yaml')).server.port)")
for old in .env .env.example updater.sh config/gestures.yaml config/bindings.yaml; do
    [[ -e "$old" ]] && say "$old belongs to version 2 and isn't read any more: delete it when all is well"
done
cat <<EOF

ORION hands is running: its MCP endpoint is http://$(hostname).local:$port/mcp
  The token Orion sends (Authorization: Bearer ...):  $(pwd)/secrets/hands_token
  Gestures as they happen:                            docker compose logs -f tracker
  Your settings (over the defaults):                  $(pwd)/config/tracker.yaml
EOF
