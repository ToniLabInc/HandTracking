"""ORION hands as an MCP server: the tracker itself serves it over Streamable HTTP (``server.path``, /mcp). Orion
connects to it as mcp_servers.hands (its url and the bearer token); so can any MCP client.

Resources (all application/json):

- ``orion://intents``: ``{"intents": [...], "instance", "now"}``, the last ``intents.keep`` gestures, oldest first,
  each ``{"id", "intent": "gesture", "gesture", "args": {"kind", "phase", "held", "showing"}, "hand", "score",
  "ts"}``. Ids only increase (milliseconds of the Pi's clock). ``instance`` names this run of the tracker (it changes
  when the tracker restarts, so a client starts over), ``now`` is the Pi's clock as it was read: an entry's age is
  ``now - id``, whatever the client's own clock says. A hand shape is reported as it starts (``phase: start``), then
  while it is held, at each hold step (``phase: hold``, ``held`` in seconds, ``showing`` the id of its start). What a
  gesture does is the client's to decide (Orion: Settings, its MCP server).
- ``orion://gestures``: the vocabulary, ``{"gestures": [{"name", "title", "kind", "how"}], "static_hold",
  "hold_steps", "max_hold", "two_hands": {"together", "apart"}, "tracker"}`` (``static_hold``: how long a shape is held
  before it counts; ``two_hands``: the distances, in palm lengths, that make hands_together and hands_apart).
- ``orion://hands``: the live view, its latest frame: ``{"seq", "ts", "w", "h", "fps", "mode", "hands": [{"track",
  "hand", "shape", "score", "held", "pts"}], "apart", "link": {...}}`` (``pts``: the 21 landmarks, x and y from 0 to 1
  in the mirrored image; ``apart``: how far apart two hands are, in palm lengths, null without two).
- ``orion://hands/camera``: the same, with the camera's latest picture (``jpg``: a small JPEG, base64; ``jpg_seq``:
  the frame it came with, so a reader decodes each picture once).

Each change sends ``notifications/resources/updated`` on the ``subscriptions/listen`` streams (MCP 2026-07-28) that
asked for it: each gesture, and each live-view frame (``live_view.fps`` a second at most; a client reads the latest
when it can). An open stream is what keeps the camera on (a stream on any resource), and the live view (on
orion://hands; on orion://hands/camera for the picture too). Reading a live view also keeps it on for
``live_view.lease_s``, for a client that reads without listening.

Tools: hands_status, hands_gestures, hands_pause, hands_resume (and hands_inject, with debug.inject_tool).

The endpoint answers LAN clients only (``server.lan_only``), and every request but ``GET /health`` (Docker's health
check) must carry the bearer token of ``server.token_file``.
"""

import asyncio
import contextlib
import hmac
import ipaddress
import json
import logging
import math
import secrets
import signal
import socket
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, Final

import mcp_types as types
import uvicorn
from mcp.server import Server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ResourceUpdated
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from handtracker import VERSION, config
from handtracker.config import TrackerConfig
from handtracker.gestures import Engine, Event
from handtracker.service import CameraSwitch, Pipeline, Stats, View, rss_mb
from handtracker.vocabulary import GESTURES, HOLD_STEPS, describe

log = logging.getLogger("handtracker.server")
events_log = logging.getLogger("handtracker.events")

INTENTS: Final = "orion://intents"
VOCABULARY: Final = "orion://gestures"
LIVE: Final = "orion://hands"
CAMERA: Final = "orion://hands/camera"
VIEWS: Final = (LIVE, CAMERA)
WATCH_EVERY_S: Final = 2.0  # how often tracker.yaml is looked at
TICK_S: Final = 1.0  # the camera switch and the live view's leases are brought up to date at least this often
MIN_TOKEN: Final = 16  # characters a bearer token needs at least
PAUSE_MAX_MIN: Final = 1440  # a timed pause lasts a day at most
COMPACT: Final = json.JSONEncoder(separators=(",", ":"))  # the resources' JSON, without spaces

INSTRUCTIONS: Final = (
    "Hand gestures from the ORION hands tracker (a camera on a Raspberry Pi; video never leaves it, except a small "
    "picture for the live view when asked). Listen to the resource orion://intents (subscriptions/listen): each "
    "update means new gestures. Read it and act, in id order, on every entry whose id is above the last one you "
    "handled (ids only increase within an instance); skip entries older than about 10 seconds (now - id). Each entry "
    "is a gesture (intent: gesture): a hand shape starts (args.phase start) and, held on, is reported again at its "
    "hold steps (phase hold, args.held in seconds, args.showing the id of its start); motions are reported once. "
    "What a gesture does is up to you. orion://gestures lists every gesture and how to make it; orion://hands is a "
    "live view of the hands (listen to it, or read it, to keep it on), orion://hands/camera the same with the "
    "camera's picture. Tools: hands_status (is the camera working, hands in view), hands_gestures (the gestures and "
    "how to make them), hands_pause / hands_resume (gestures off and on; the camera too).")


def _tool(name: str, title: str, description: str, properties: dict[str, Any] | None = None,
          required: list[str] | None = None, *, read_only: bool = False, idempotent: bool = True) -> types.Tool:
    """A tool definition."""
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}, "additionalProperties": False}
    if required:
        schema["required"] = required
    return types.Tool(name=name, title=title, description=description, input_schema=schema,
                      annotations=types.ToolAnnotations(read_only_hint=read_only, destructive_hint=False,
                                                        idempotent_hint=idempotent, open_world_hint=False))


TOOLS: Final = (
    _tool("hands_status", "Hand-gesture status",
          "Whether the hand-gesture tracker is working: camera, frame rate, hands in view, whether gestures are "
          "paused, the live view, and the last gesture seen.", read_only=True),
    _tool("hands_gestures", "The hand gestures",
          "Every hand gesture the camera recognises, and how to make it (what each does is set in Orion's settings, "
          "under this MCP server).", read_only=True),
    _tool("hands_pause", "Pause gestures",
          "Turn hand gestures off (the camera too) until hands_resume, or for some minutes.",
          {"minutes": {"type": "number", "exclusiveMinimum": 0, "maximum": PAUSE_MAX_MIN,
                       "description": "resume by itself after this many minutes (leave out: until resumed)"}}),
    _tool("hands_resume", "Resume gestures", "Turn hand gestures back on."),
)
INJECT: Final = _tool("hands_inject", "Make a gesture (testing)",
                      "Report a gesture as if the camera had seen it (debug.inject_tool; for tests only).",
                      {"gesture": {"type": "string", "enum": list(GESTURES)}}, ["gesture"], idempotent=False)
RESOURCES: Final = (
    types.Resource(uri=INTENTS, name="intents", title="Gestures", mime_type="application/json",
                   description="The last gestures, oldest first: {intents: [{id, intent: gesture, gesture, args: "
                               "{kind, phase, held, showing}, hand, score, ts}], instance, now}. Listen: every new "
                               "one sends notifications/resources/updated."),
    types.Resource(uri=VOCABULARY, name="gestures", title="The gesture vocabulary", mime_type="application/json",
                   description="Every gesture the tracker recognises: {gestures: [{name, title, kind, how}], "
                               "hold_steps, max_hold, tracker}."),
    types.Resource(uri=LIVE, name="hands", title="Live view", mime_type="application/json",
                   description="The latest frame: each hand's landmarks, shape and how long it has held it. Listen "
                               "to it (or read it) to keep the view on."),
    types.Resource(uri=CAMERA, name="camera", title="Live view with the picture", mime_type="application/json",
                   description="The latest frame with the camera's picture (a small JPEG, base64, in jpg)."),
)


class Tracker:
    """What the endpoint serves and who follows it: the gestures feed, the live view, the state of the camera.

    Args:
        cfg: the settings (the server's own parts are read once; the engine takes reloads).
        engine: the gesture engine.
        stats: the camera loop's measurements.
        view: who watches the live view (this sets the counts; the camera loop reads them).
    """

    def __init__(self, cfg: TrackerConfig, engine: Engine, stats: Stats, view: View) -> None:
        self.cfg, self.engine, self.stats, self.view = cfg, engine, stats, view
        self.feed: deque[dict[str, Any]] = deque(maxlen=cfg.intents.keep)
        self.instance = secrets.token_hex(6)
        self.frame: Event | None = None  # the latest live-view frame, without its picture...
        self.picture: tuple[int, str] | None = None  # ...which is kept apart: (its frame's seq, the JPEG in base64)
        self.leases = dict.fromkeys(VIEWS, -math.inf)  # monotonic: each view is wanted until then, read or not
        self.listening: Counter[str] = Counter()  # open listen streams, per resource
        self.bus = InMemorySubscriptionBus()
        self.started = time.monotonic()
        self.cpu_percent = 0.0
        self.poke = asyncio.Event()  # set when the camera may be wanted, or not any more
        self.switch: CameraSwitch | None = None  # the camera's (set by serve)
        self._listen = ListenHandler(self.bus, max_subscriptions=cfg.server.max_listeners)
        self.tools = [*TOOLS, INJECT] if cfg.debug.inject_tool else list(TOOLS)
        self.server: Server[Any] = Server(
            "orion-hands", version=VERSION, title="ORION hands",
            description="Hand gestures from the camera on the Raspberry Pi, and a live view of the hands",
            instructions=INSTRUCTIONS,
            on_list_tools=lambda _ctx, _params: self._listed(types.ListToolsResult(tools=self.tools)),
            on_call_tool=self._call_tool,
            on_list_resources=lambda _ctx, _params: self._listed(types.ListResourcesResult(resources=list(RESOURCES))),
            on_read_resource=self._read_resource,
            on_subscriptions_listen=self._listened)

    @staticmethod
    async def _listed[T](result: T) -> T:
        return result

    # ------------------------------------------------------------------------------------------------ demand
    def demand(self) -> int:
        """How many follow the tracker: open listen streams, and live views read lately."""
        now = time.monotonic()
        return sum(self.listening.values()) + sum(lease > now for lease in self.leases.values())

    def refresh_view(self) -> None:
        """Bring the live view in line with who follows it (and forget its frames once nobody does)."""
        now = time.monotonic()
        images = self.listening[CAMERA] + (self.leases[CAMERA] > now)
        landmarks = self.listening[LIVE] + (self.leases[LIVE] > now) + images  # the picture comes with the hands
        if (landmarks > 0, images > 0) != (self.view.landmarks > 0, self.view.images > 0):
            log.info("live view: %s", "picture" if images else "landmarks" if landmarks else "off")
        self.view.landmarks, self.view.images = landmarks, images
        if not landmarks:
            self.frame = None
        if not images:
            self.picture = None

    async def _listened(self, ctx: Any, params: types.SubscriptionsListenRequestParams) -> Any:
        """A subscriptions/listen stream, counted for as long as it stays open."""
        uris = [str(uri) for uri in params.notifications.resource_subscriptions or ()]
        self.listening.update(uris)
        self.refresh_view()
        self.poke.set()  # the camera turns on
        log.info("a client listens to %s (%d stream(s) open)", ", ".join(uris) or "nothing",
                 sum(self.listening.values()))
        try:
            return await self._listen(ctx, params)
        finally:
            self.listening -= Counter(uris)  # (a count down to zero is dropped)
            self.refresh_view()
            self.poke.set()  # the last one gone: the camera turns off after the grace
            log.info("a listen stream closed (%d open)", sum(self.listening.values()))

    def close_streams(self) -> None:
        """End every listen stream (shutting down)."""
        self._listen.close()

    # ------------------------------------------------------------------------------------------------ events
    async def on_events(self, events: list[Event]) -> None:
        """A frame's events from the camera loop: log them, and add the gestures to the feed."""
        added = False
        for event in events:
            _log_event(event)
            if event["type"] != "gesture":
                continue
            args = {key: event[key] for key in ("kind", "phase", "held", "showing") if key in event}
            self.feed.append({"id": event["id"], "intent": "gesture", "gesture": event["gesture"], "args": args,
                              "hand": event["hand"], "score": event["score"], "ts": event["ts"]})
            added = True
        if added:
            await self.bus.publish(ResourceUpdated(uri=INTENTS))

    async def on_frame(self, frame: Event) -> None:
        """A live-view frame: keep it (its picture apart), and tell the streams listening to each view."""
        if "jpg" in frame:
            self.picture = (frame["seq"], frame.pop("jpg"))
        self.frame = frame
        for uri in VIEWS:
            if self.listening[uri]:
                await self.bus.publish(ResourceUpdated(uri=uri))

    # ------------------------------------------------------------------------------------------------ resources
    def feed_text(self) -> str:
        """orion://intents."""
        return COMPACT.encode({"intents": list(self.feed), "instance": self.instance, "now": int(time.time() * 1000)})

    def vocabulary(self) -> dict[str, Any]:
        """orion://gestures."""
        r = self.engine.config.recognition
        return {"gestures": describe(), "static_hold": r.static_hold, "hold_steps": list(HOLD_STEPS),
                "max_hold": r.max_hold, "two_hands": {"together": r.together_distance, "apart": r.apart_distance},
                "tracker": VERSION}

    def view_text(self, uri: str) -> str:
        """A live view: the latest frame (the camera's view: with the latest picture), and how the tracker stands."""
        frame = dict(self.frame or {"seq": 0, "hands": []})
        if uri == CAMERA and self.picture is not None:
            frame["jpg_seq"], frame["jpg"] = self.picture
        frame["link"] = {"connected": True, "error": "", "camera": self.camera_ok(),
                         "paused": self.engine.paused(time.monotonic()), "tracker": VERSION}
        return COMPACT.encode(frame)

    async def _read_resource(self, _ctx: Any, params: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        uri = str(params.uri)
        if uri == INTENTS:
            text = self.feed_text()
        elif uri == VOCABULARY:
            text = COMPACT.encode(self.vocabulary())
        elif uri in VIEWS:
            self.leases[uri] = time.monotonic() + self.cfg.live_view.lease_s
            self.refresh_view()
            self.poke.set()
            text = self.view_text(uri)
        else:
            raise MCPError(code=types.INVALID_PARAMS,
                           message=f"unknown resource {uri}; there are {', '.join(str(r.uri) for r in RESOURCES)}")
        return types.ReadResourceResult(contents=[types.TextResourceContents(uri=uri, mime_type="application/json",
                                                                             text=text)])

    # ------------------------------------------------------------------------------------------------ tools
    async def _call_tool(self, _ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        try:
            match params.name:
                case "hands_status":
                    text = self.status_text()
                case "hands_gestures":
                    text = gestures_text(self.vocabulary())
                case "hands_pause":
                    text = self.pause(args.get("minutes"))
                case "hands_resume":
                    self.engine.resume()
                    self.poke.set()
                    log.info("gestures resumed")
                    text = "Hand gestures are back on."
                case "hands_inject" if self.cfg.debug.inject_tool:
                    gesture = args.get("gesture")
                    if gesture not in GESTURES:
                        raise ValueError(f"gesture must be one of {', '.join(GESTURES)}")
                    await self.on_events(self.engine.inject(gesture))
                    text = f"Reported {gesture}."
                case _:
                    raise ValueError(f"there is no tool called {params.name}")
        except ValueError as e:
            return types.CallToolResult(content=[types.TextContent(type="text", text=f"Error: {e}")], is_error=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    def pause(self, minutes: Any) -> str:
        """hands_pause: gestures (and the camera) off, until resumed or for ``minutes``."""
        if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, int | float)
                                    or not 0 < minutes <= PAUSE_MAX_MIN):
            raise ValueError(f"minutes must be a number above 0 and at most {PAUSE_MAX_MIN}")
        self.engine.pause(time.monotonic(), minutes * 60 if minutes else None)
        self.poke.set()  # paused, the camera turns off
        log.info("gestures paused%s", f" for {minutes:g} min" if minutes else "")
        return (f"Hand gestures are paused for {minutes:g} minutes." if minutes
                else "Hand gestures are paused until resumed.")

    # ------------------------------------------------------------------------------------------------ state
    def camera_on(self) -> bool:
        """Whether the camera loop runs (only on demand)."""
        return self.switch is not None and self.switch.on

    def camera_ok(self) -> bool:
        """Whether frames arrive."""
        return self.stats.camera and time.monotonic() - self.stats.last_frame <= self.cfg.camera.fresh_s

    def health(self) -> dict[str, Any]:
        """GET /health: ok unless the camera should deliver frames and doesn't (past its warm-up)."""
        now = time.monotonic()
        active = self.camera_on()
        since = self.switch.since if self.switch is not None else -math.inf
        ok = self.camera_ok() or not active or now - since < self.cfg.camera.warmup_s
        snap = self.engine.snapshot(now)
        return {"ok": ok, "camera": self.camera_ok(), "active": active, "clients": self.demand(),
                "fps": self.stats.rates(now)[0] if active else 0.0, "inference_ms": round(self.stats.inference_ms, 1),
                "hands": snap["hands"], "paused": snap["paused"], "version": VERSION}

    def state(self) -> dict[str, Any]:
        """Everything /health says, and the timings, the last gesture, the live view, CPU and memory."""
        now = time.monotonic()
        fps, capture_fps = self.stats.rates(now)
        switch = self.switch
        mode = (switch.pipeline.scheduler.mode if switch is not None and switch.on and switch.pipeline.scheduler
                else "starting" if self.camera_on() else "off")
        return {**self.health(), **self.engine.snapshot(now), "fps": fps, "capture_fps": capture_fps,
                "pipeline_ms": round(self.stats.pipeline_ms, 1), "camera_format": self.stats.camera_format,
                "frames": self.stats.frames, "mode": mode, "uptime": round(now - self.started),
                "cpu_percent": self.cpu_percent, "rss_mb": rss_mb(), "streams": sum(self.listening.values()),
                "view": {"landmarks": self.view.landmarks, "images": self.view.images}}

    def status_text(self) -> str:
        """hands_status, in words."""
        live = self.state()
        last = live["last_gesture"]
        paused_for = live["paused_for"]
        return "\n".join([
            camera_line(live),
            f"Hands in view: {live['hands']}.",
            "Gestures: " + (("paused" + (f" for {paused_for:g} s more" if paused_for else " until resumed"))
                            if live["paused"] else "on") + ".",
            f"Live view: {'on' if live['view']['landmarks'] else 'off'}; clients following: {live['streams']}.",
            "Last gesture: " + (f"{last['gesture']} ({last['hand']}, {last['score']}) {ago(last['ts'])}" if last
                                else "none yet") + ".",
            f"Tracker {VERSION}, CPU {live['cpu_percent']}% of a core, memory {live['rss_mb']} MB, "
            f"up {duration(live['uptime'])}.",
        ])


def _log_event(event: Event) -> None:
    """One line per gesture and change, for `docker compose logs -f` (the hold steps of a shape only in debug)."""
    if event["type"] == "gesture":
        if event["phase"] == "start":
            events_log.info("gesture %s (%s, %.2f)", event["gesture"], event["hand"], event["score"])
        else:
            events_log.debug("gesture %s held %.1f s", event["gesture"], event["held"])
    elif event["type"] == "presence":
        events_log.info("hands in view: %d", event["hands"])


KINDS: Final = {"shape": "Hand shapes (held still)", "motion": "Motions of one hand", "two_hands": "Both hands"}


def gestures_text(vocabulary: dict[str, Any]) -> str:
    """hands_gestures: the gestures by kind, each with how to make it."""
    lines = []
    for kind, title in KINDS.items():
        lines.append(f"{title}:")
        lines += [f"- {g['name']} ({g['title']}): {g['how']}" for g in vocabulary["gestures"] if g["kind"] == kind]
    lines.append("What each gesture does is set in Orion's settings, under this MCP server.")
    return "\n".join(lines)


def camera_line(live: dict[str, Any]) -> str:
    """The camera in words: working, off (it is on only while a client follows and gestures aren't paused), starting,
    or on but delivering nothing."""
    if live["camera"]:
        return (f"Camera: working, {live['fps']} frames/s processed (camera {live['capture_fps']}), model "
                f"{live['inference_ms']} ms per frame.")
    if not live["active"]:
        why = "gestures are paused" if live["paused"] else "no client is following"
        return f"Camera: off ({why}; it turns on by itself when needed, and its light is off)."
    return "Camera: starting, or NOT WORKING if this lasts (no frames yet)."


def ago(ts: float) -> str:
    """How long ago a unix time was, in words."""
    return duration(time.time() - ts) + " ago"


def duration(seconds: float) -> str:
    """A duration in words (12 s, 3 min, 2 h)."""
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{round(seconds / 3600, 1):g} h"


# ---------------------------------------------------------------------------------------------------- the endpoint
def lan(address: str | None) -> bool:
    """Whether a client address is loopback, private or link-local (the LAN)."""
    try:
        ip = ipaddress.ip_address(address or "")
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_private or ip.is_loopback or ip.is_link_local


class Guard:
    """LAN clients only (``server.lan_only``), and the bearer token on every request but GET /health (pure ASGI, so
    the endpoint's streams pass through untouched).

    Args:
        app: the endpoint.
        token: the token.
        lan_only: refuse clients outside the LAN.
    """

    def __init__(self, app: ASGIApp, token: str, *, lan_only: bool) -> None:
        self.app, self.token, self.lan_only = app, token.encode(), lan_only

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            client = scope.get("client")
            if self.lan_only and not lan(client[0] if client else None):
                await JSONResponse({"error": "LAN clients only"}, 403)(scope, receive, send)
                return
            if scope["path"] != "/health" and not self._authorized(scope):
                await JSONResponse({"error": "missing or wrong token"}, 401,
                                   headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        await self.app(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        """Whether the request carries the token."""
        header = dict(scope["headers"]).get(b"authorization", b"")
        given = header[7:].strip() if header[:7].lower() == b"bearer " else b""
        return bool(given) and hmac.compare_digest(given, self.token)


def build_app(tracker: Tracker, token: str) -> ASGIApp:
    """The endpoint: MCP at server.path, /health, behind the guard."""
    s = tracker.cfg.server

    async def health(_: Request) -> JSONResponse:
        body = tracker.health()
        return JSONResponse(body, 200 if body["ok"] else 503)

    app = tracker.server.streamable_http_app(
        streamable_http_path=s.path, host=s.host,
        # the guard checks every client; the Pi is reached by more than one name (raspberrypi.local, its address)
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        custom_starlette_routes=[Route("/health", health, methods=["GET"])])
    return Guard(app, token, lan_only=s.lan_only)


def read_token(path: Path) -> str:
    """The bearer token, from its file (one line).

    Raises:
        config.ConfigError: it is missing, unreadable or too short.
    """
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise config.ConfigError(f"server.token_file {path}: {e.strerror or e} (install-pi.sh makes it)") from None
    if len(token) < MIN_TOKEN:
        raise config.ConfigError(f"server.token_file {path}: the token must be at least {MIN_TOKEN} characters")
    return token


def listening_socket(host: str, port: int, dead_after_s: float) -> socket.socket:
    """The endpoint's socket, with keepalives and a send timeout that its connections inherit: a client that vanished
    without closing (its PC switched off) is noticed within ``dead_after_s``, and the camera it kept on goes off."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, max(1, round(dead_after_s / 3)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, max(1, round(dead_after_s / 6)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, round(dead_after_s * 1000))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # a reply's last small write isn't held for an ACK
    sock.bind((host, port))
    sock.listen(128)
    return sock


# ---------------------------------------------------------------------------------------------------- serving
async def serve(cfg: TrackerConfig, path: Path) -> int:
    """Run the tracker until SIGTERM/SIGINT (0), or until the camera stays unusable (1, so Docker restarts it).

    The endpoint is always up; the camera only on demand (service.CameraSwitch): while a client follows and gesture
    control isn't paused.
    """
    token = read_token(cfg.server.token_file)
    c, p = cfg.camera, cfg.processing
    log.info("ORION hands tracker %s: camera %s %dx%d@%d %s, processing %g fps (idle %g, deep idle %g after %g s), "
             "live view %g fps (picture %g), model input width %s, mirror %s, camera %s", VERSION, c.device, c.width,
             c.height, c.fps, c.fourcc, p.fps, p.idle_fps, p.deep_idle_fps, p.deep_idle_after_s, cfg.live_view.fps,
             cfg.live_view.picture_fps, p.infer_width or "as captured", "on" if c.mirror else "off",
             "ALWAYS ON" if c.always_on else f"on demand (off {c.idle_grace_s:g} s after the last client)")
    engine = Engine(cfg)
    stats = Stats()
    view = View(cfg.live_view.fps, cfg.live_view.picture_fps)
    tracker = Tracker(cfg, engine, stats, view)
    loop = asyncio.get_running_loop()
    pending: set[asyncio.Task[None]] = set()

    def later(job: Callable[[], Awaitable[None]]) -> None:
        """Run a coroutine function on the event loop, from the camera thread (in order; kept until it ends)."""

        def start() -> None:
            task = loop.create_task(job())
            pending.add(task)
            task.add_done_callback(pending.discard)

        loop.call_soon_threadsafe(start)

    def pipeline() -> Pipeline:
        return Pipeline(cfg, engine, lambda events: later(lambda: tracker.on_events(events)), stats, view=view,
                        publish_frame=lambda frame: later(lambda: tracker.on_frame(frame)))

    switch = tracker.switch = CameraSwitch(pipeline, c.idle_grace_s)
    sock = listening_socket(cfg.server.host, cfg.server.port, cfg.server.dead_client_after_s)
    endpoint = Endpoint(uvicorn.Config(build_app(tracker, token), log_level="warning", access_log=False,
                                       proxy_headers=False, lifespan="on"))
    stop = asyncio.Event()

    def stopping() -> None:
        stop.set()
        tracker.poke.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping)
    serving = loop.create_task(endpoint.serve(sockets=[sock]), name="endpoint")
    background = [loop.create_task(_status_loop(tracker), name="status"),
                  loop.create_task(_watch_loop(tracker, path), name="watch")]
    log.info("MCP endpoint on http://%s:%d%s (the bearer token required; /health without)", cfg.server.host,
             cfg.server.port, cfg.server.path)
    code = 0
    try:
        while not stop.is_set() and not serving.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(tracker.poke.wait(), TICK_S)
            tracker.poke.clear()
            tracker.refresh_view()  # leases run out
            now = time.monotonic()
            paused = engine.paused(now)
            wanted = (c.always_on or tracker.demand() > 0) and not paused
            change = switch.tick(wanted, now, at_once=paused)
            if change == "on":
                log.info("camera on (%d following)", tracker.demand())
            elif change == "off":
                log.info("camera off (%s)", "gestures paused" if paused else "nobody following")
                await tracker.on_events(engine.rest())
            if switch.failed:
                log.error("the camera stayed unusable: %s; exiting so Docker restarts the tracker", switch.failed)
                code = 1
                break
        if serving.done() and serving.exception() is not None:
            log.error("the endpoint stopped: %s", serving.exception())
            code = 1
    finally:
        switch.tick(False, time.monotonic(), at_once=True)
        tracker.close_streams()
        endpoint.should_exit = True
        for task in background:
            task.cancel()
        await asyncio.gather(serving, *background, return_exceptions=True)
        for thread in (switch.closing, switch.pipeline):
            if thread is not None:
                await asyncio.to_thread(thread.join, 3.0)
    log.info("stopped")
    return code


class Endpoint(uvicorn.Server):
    """uvicorn's server, leaving the signals to ``serve`` (which stops the camera first)."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


async def _status_loop(tracker: Tracker) -> None:
    """Measure CPU every status.every_s, and log the state every status.log_every_s."""
    s = tracker.cfg.status
    tracker.stats.cpu_percent()
    logged = time.monotonic()
    while True:
        await asyncio.sleep(s.every_s)
        tracker.cpu_percent = tracker.stats.cpu_percent()
        if time.monotonic() - logged < s.log_every_s:
            continue
        logged = time.monotonic()
        state = tracker.state()
        camera = "ok" if state["camera"] else "DOWN" if state["active"] else "off"
        log.info("status: camera %s, %.1f fps (camera %.1f), model %.1f ms, pipeline %.1f ms, cpu %.0f%%, rss %.0f MB, "
                 "hands %d, streams %d, watching %d", camera, state["fps"], state["capture_fps"],
                 state["inference_ms"], state["pipeline_ms"], state["cpu_percent"], state["rss_mb"], state["hands"],
                 state["streams"], state["view"]["landmarks"])


async def _watch_loop(tracker: Tracker, path: Path) -> None:
    """Reload tracker.yaml when it changes: recognition and ignore apply at once; the rest is said to need a restart.
    A broken edit keeps the last good settings."""
    watcher = config.Watcher(path)
    while True:
        await asyncio.sleep(WATCH_EVERY_S)
        if not watcher.changed():
            continue
        try:
            new = await asyncio.to_thread(config.load, path)
        except (OSError, config.ConfigError) as e:
            log.error("%s not reloaded, keeping the settings in use: %s", path, e)
            continue
        if later := new.restart_needed(tracker.cfg):
            log.warning("%s: %s apply after a restart (docker compose restart tracker)", path, ", ".join(later))
        tracker.engine.configure(new)
        log.info("%s reloaded: recognition and ignore apply now (%d gesture(s) ignored)", path, len(new.ignore))
