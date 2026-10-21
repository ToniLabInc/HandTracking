#!/usr/bin/env bash
# Deploy this checkout to the Raspberry Pi from the PC: copy tracker/ and deploy/ over SSH, then run install-pi.sh
# there (it builds the image on the Pi and recreates the tracker; the Pi's config/, secrets/ and
# compose.override.yaml stay). The Pi is named in deploy/pi.yaml (copy deploy/pi.example.yaml).
#
#   deploy/push-pi.sh              # deploy
#   deploy/push-pi.sh --rollback   # back to the image that ran before the last deploy
set -euo pipefail

HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
SETTINGS=$HERE/deploy/pi.yaml
if [[ ! -f "$SETTINGS" ]]; then
    echo "deploy/pi.yaml is missing: copy deploy/pi.example.yaml to it and fill it in" >&2
    exit 2
fi
# pi.yaml is a flat mapping: "name: value  # comment"
value() { sed -n "s/^$1:[[:space:]]*//p" "$SETTINGS" | sed 's/[[:space:]]*#.*$//; s/^"\(.*\)"$/\1/' | head -n 1; }
PI=$(value host)
KEY=$(value key)
DIR=$(value dir)
for name in host key dir; do
    if [[ -z "$(value "$name")" ]]; then
        echo "deploy/pi.yaml: set $name" >&2
        exit 2
    fi
done
KEY=${KEY/#\~/$HOME}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$KEY")

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
on_pi() { "${SSH[@]}" "$PI" "$@"; }

if [[ "${1:-}" == "--rollback" ]]; then
    say "rolling the tracker on $PI back to the image that ran before the last deploy"
    on_pi "cd ~/$DIR/deploy && docker image inspect orion-hands:previous >/dev/null \
           && docker tag orion-hands:previous orion-hands:latest && docker compose up -d --wait"
    exit 0
fi

say "copying tracker/ and deploy/ to $PI:~/$DIR (its config/, secrets/ and compose.override.yaml stay)"
rsync -az --delete -e "${SSH[*]}" \
    --exclude .venv/ --exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ \
    "$HERE/tracker/" "$PI:$DIR/tracker/"
rsync -az -e "${SSH[*]}" --exclude config/ --exclude secrets/ --exclude compose.override.yaml --exclude pi.yaml \
    --exclude .env "$HERE/deploy/" "$PI:$DIR/deploy/"
build=$(git -C "$HERE" rev-parse --short=12 HEAD)
git -C "$HERE" diff --quiet HEAD -- tracker || build="$build-dirty"

say "installing on the Pi (build $build)"
on_pi "cd ~/$DIR/deploy && BUILD=$build ./install-pi.sh"
