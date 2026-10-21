# syntax=docker/dockerfile:1.7
# ORION hands tracker: a USB camera, MediaPipe's gesture recognizer, gestures for Orion as an MCP server on the LAN.
# Built on the Pi by deploy/install-pi.sh (linux/arm64); CI builds linux/arm64 and linux/amd64 to check it.

ARG PYTHON_IMAGE=python:3.13.16-slim-trixie

# ---------------------------------------------------------------------------------------------------- build
FROM ${PYTHON_IMAGE} AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
# binutils to strip the libraries. MediaPipe's library links libEGL and libGLESv2 (for its GPU path, unused here):
# Debian's packages for them pull in all of Mesa (200 MB), so only glvnd's small loader libraries are taken.
RUN apt-get update && apt-get install -y --no-install-recommends binutils \
 && mkdir -p /tmp/gl /opt/gl && cd /tmp/gl && apt-get download libegl1 libgles2 libglvnd0 \
 && for deb in *.deb; do dpkg-deb -x "$deb" .; done \
 && cp -a usr/lib/*-linux-gnu/libEGL.so.1* usr/lib/*-linux-gnu/libGLESv2.so.2* usr/lib/*-linux-gnu/libGLdispatch.so.0* \
       /opt/gl/ \
 && cd / && rm -rf /tmp/gl /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
# Exact versions with hashes, no dependency resolution (see requirements.txt); then trim: MediaPipe's own tests,
# debug symbols (about 40 MB), and precompile the bytecode (the container's filesystem is read-only).
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-deps --require-hashes --only-binary=:all: -r /tmp/requirements.txt \
 && site=$(/opt/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && rm -rf "$site/mediapipe/tasks/python/test" "$site/mediapipe/tasks/python/benchmark" \
 && find "$site/numpy" -type d -name tests -prune -exec rm -rf {} + \
 && find "$site" -name "*.so*" -type f -exec strip --strip-unneeded {} + \
 && /opt/venv/bin/pip uninstall -y -q pip \
 && /opt/venv/bin/python -m compileall -q -j 0 "$site"
# The model, fetched at build time and checked (Apache 2.0, from Google's MediaPipe model storage).
ADD --checksum=sha256:97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482 \
    https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task \
    /opt/models/gesture_recognizer.task
RUN chmod 0644 /opt/models/gesture_recognizer.task

# ---------------------------------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE}
RUN groupadd --system --gid 10001 hands \
 && useradd --system --uid 10001 --gid hands --no-create-home --home-dir /tmp --shell /usr/sbin/nologin hands
COPY --from=build /opt/gl /usr/local/lib/glvnd
RUN echo /usr/local/lib/glvnd > /etc/ld.so.conf.d/glvnd.conf && ldconfig
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/models /opt/models
COPY handtracker /app/handtracker
RUN python -m compileall -q /app/handtracker

ARG BUILD=""
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    HANDS_BUILD=$BUILD
LABEL org.opencontainers.image.title="ORION hands tracker" \
      org.opencontainers.image.description="Hand gestures from a USB camera, as intents for the ORION assistant" \
      org.opencontainers.image.source="https://github.com/ToniLabInc/HandTracking"
WORKDIR /app
USER 10001:10001
# The Pi's settings are mounted at /config/tracker.yaml (laid over handtracker/defaults.yaml); the token at
# /run/secrets/hands_token (server.token_file).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-m", "handtracker", "health", "--config", "/config/tracker.yaml"]
ENTRYPOINT ["python", "-m", "handtracker"]
CMD ["serve", "--config", "/config/tracker.yaml"]
