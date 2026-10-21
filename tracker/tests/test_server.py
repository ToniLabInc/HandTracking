"""The MCP endpoint over real HTTP, as Orion uses it: the guard (the token, the LAN), tools and resources, gestures
reaching the clients that listen, the live view following its listeners, and the read lease."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from handtracker.config import TrackerConfig
from handtracker.gestures import Engine
from handtracker.server import CAMERA, INTENTS, LIVE, VOCABULARY, Endpoint, Tracker, build_app, lan, listening_socket
from handtracker.service import Stats, View

TOKEN = "a-test-token-long-enough"


class Running:
    """A tracker's endpoint served on a free local port (no camera)."""

    def __init__(self, tracker: Tracker, url: str) -> None:
        self.tracker, self.url = tracker, url

    @asynccontextmanager
    async def client(self, token: str = TOKEN) -> AsyncIterator[Client]:
        """An MCP client connected the way Orion connects (Streamable HTTP, the bearer token, the default mode)."""
        async with AsyncExitStack() as stack:
            http = await stack.enter_async_context(create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}))
            yield await stack.enter_async_context(Client(streamable_http_client(f"{self.url}/mcp", http_client=http)))


@pytest.fixture
def settings(cfg: TrackerConfig, tmp_path: Path) -> TrackerConfig:
    token = tmp_path / "token"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    return cfg.model_copy(update={"server": cfg.server.model_copy(update={"token_file": token, "host": "127.0.0.1"}),
                                  "debug": cfg.debug.model_copy(update={"inject_tool": True})})


@pytest.fixture
async def running(settings: TrackerConfig) -> AsyncIterator[Running]:
    tracker = Tracker(settings, Engine(settings), Stats(), View(settings.live_view.fps, settings.live_view.picture_fps))
    sock = listening_socket("127.0.0.1", 0, settings.server.dead_client_after_s)
    endpoint = Endpoint(uvicorn.Config(build_app(tracker, TOKEN), log_level="warning", lifespan="on"))
    task = asyncio.create_task(endpoint.serve(sockets=[sock]))
    await until(lambda: endpoint.started)
    yield Running(tracker, f"http://127.0.0.1:{sock.getsockname()[1]}")
    tracker.close_streams()
    endpoint.should_exit = True
    await task


async def until(check: Any, within: float = 5.0) -> None:
    """Wait for ``check()`` to be true."""
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, "it never happened"
        await asyncio.sleep(0.02)


def read(result: Any) -> dict[str, Any]:
    return json.loads(result.contents[0].text)


async def test_every_request_but_health_needs_the_token(running: Running) -> None:
    async with httpx2.AsyncClient() as http:
        health = await http.get(f"{running.url}/health")
        assert health.status_code == 200 and health.json()["version"]
        bare = await http.post(f"{running.url}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert bare.status_code == 401 and bare.headers["www-authenticate"] == "Bearer"
        wrong = await http.post(f"{running.url}/mcp", json={}, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401


def test_only_lan_addresses_are_let_in() -> None:
    assert all(map(lan, ["192.168.1.20", "10.0.0.5", "172.16.3.4", "127.0.0.1", "::1", "fe80::1%eth0",
                         "::ffff:192.168.1.20"]))
    assert not any(map(lan, ["8.8.8.8", "2001:4860:4860::8888", "", None, "not an address"]))


async def test_tools_resources_and_the_modern_protocol(running: Running) -> None:
    async with running.client() as client:
        assert client.protocol_version == "2026-07-28"  # Orion listens on this protocol
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"hands_status", "hands_gestures", "hands_pause", "hands_resume", "hands_inject"}
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        assert uris == {INTENTS, VOCABULARY, LIVE, CAMERA}
        vocabulary = read(await client.read_resource(VOCABULARY))
        assert len(vocabulary["gestures"]) == 22 and vocabulary["hold_steps"] == [0.6, 1.0, 1.5, 2.0]
        assert vocabulary["max_hold"] == 8.0 and vocabulary["two_hands"] == {"together": 1.8, "apart": 3.0}
        assert vocabulary["static_hold"] == 0.3
        status = (await client.call_tool("hands_status", {})).content[0].text
        assert "Camera: off" in status and "Gestures: on." in status


async def test_a_gesture_reaches_the_clients_that_listen(running: Running) -> None:
    tracker = running.tracker
    async with running.client() as client, client.listen(resource_subscriptions=[INTENTS]) as sub:
        await until(lambda: tracker.demand() == 1)  # the camera's demand: one stream open
        said = (await client.call_tool("hands_inject", {"gesture": "victory"})).content[0].text
        assert said == "Reported victory."
        event = await asyncio.wait_for(anext(aiter(sub)), 5)
        assert str(event.uri) == INTENTS
        feed = read(await client.read_resource(INTENTS))
        (entry,) = feed["intents"]
        assert (entry["intent"], entry["gesture"], entry["args"]["kind"], entry["args"]["phase"]) == (
            "gesture", "victory", "shape", "start")
        assert entry["args"]["showing"] == entry["id"]  # a shape's start: its own id
        assert feed["instance"] == tracker.instance and abs(feed["now"] - time.time() * 1000) < 5000
        await client.call_tool("hands_inject", {"gesture": "swipe_left"})
        motion = read(await client.read_resource(INTENTS))["intents"][-1]
        assert motion["args"] == {"kind": "motion", "phase": "start", "held": 0.0}  # a motion has no showing
        assert motion["id"] > entry["id"]
    await until(lambda: tracker.demand() == 0)  # the stream closed with the client


async def test_the_live_view_follows_its_listeners(running: Running) -> None:
    tracker, view = running.tracker, running.tracker.view
    async with running.client() as client, client.listen(resource_subscriptions=[LIVE]) as sub:
        await until(lambda: view.landmarks == 1 and view.images == 0)
        await tracker.on_frame({"seq": 7, "ts": 1.0, "w": 640, "h": 480, "fps": 20.0, "mode": "one", "hands": []})
        assert str((await asyncio.wait_for(anext(aiter(sub)), 5)).uri) == LIVE
        frame = read(await client.read_resource(LIVE))
        assert frame["seq"] == 7 and frame["link"]["connected"] and "jpg" not in frame
        async with client.listen(resource_subscriptions=[CAMERA]):
            await until(lambda: view.images == 1 and view.landmarks >= 2)  # the picture comes with the hands
            await tracker.on_frame({"seq": 8, "ts": 1.1, "w": 640, "h": 480, "fps": 20.0, "mode": "one",
                                    "hands": [], "jpg": "/9g="})
            camera = read(await client.read_resource(CAMERA))
            assert (camera["seq"], camera["jpg_seq"], camera["jpg"]) == (8, 8, "/9g=")
    tracker.leases = dict.fromkeys(tracker.leases, 0.0)  # the reads' leases, run out
    await until(lambda: (tracker.refresh_view() or view.landmarks) == 0)
    assert tracker.frame is None and tracker.picture is None  # nobody watching: forgotten


async def test_every_frame_is_announced_even_after_one_nobody_heard(running: Running) -> None:
    """A stream is counted a moment before it can hear: a frame then is announced to nobody, and the next ones still
    reach it (no frame waits for a read)."""
    tracker = running.tracker
    frame = {"seq": 1, "ts": 1.0, "w": 640, "h": 480, "fps": 20.0, "mode": "one", "hands": []}
    tracker.listening[LIVE] += 1  # opening: counted, not subscribed yet
    await tracker.on_frame(dict(frame))
    tracker.listening[LIVE] -= 1
    async with running.client() as client, client.listen(resource_subscriptions=[LIVE]) as sub:
        await until(lambda: tracker.listening[LIVE] == 1)
        for seq in (2, 3):  # two in a row, unread: both announced
            await tracker.on_frame({**frame, "seq": seq})
        events = aiter(sub)
        assert [str((await asyncio.wait_for(anext(events), 5)).uri) for _ in range(2)] == [LIVE, LIVE]


async def test_reading_a_view_keeps_it_on_a_while(running: Running) -> None:
    tracker = running.tracker
    async with running.client() as client:
        await client.read_resource(LIVE)
        assert tracker.view.landmarks == 1 and tracker.demand() == 1
        tracker.leases[LIVE] = time.monotonic() - 1  # its lease runs out
        tracker.refresh_view()
        assert tracker.view.landmarks == 0 and tracker.demand() == 0


async def test_pause_and_resume(running: Running) -> None:
    engine = running.tracker.engine
    async with running.client() as client:
        said = (await client.call_tool("hands_pause", {"minutes": 5})).content[0].text
        assert said == "Hand gestures are paused for 5 minutes." and engine.paused(time.monotonic())
        assert (await client.call_tool("hands_resume", {})).content[0].text == "Hand gestures are back on."
        assert not engine.paused(time.monotonic())
        wrong = await client.call_tool("hands_pause", {"minutes": 0})
        assert wrong.is_error and "minutes must be" in wrong.content[0].text


def test_the_inject_tool_is_only_there_for_debugging(settings: TrackerConfig) -> None:
    plain = settings.model_copy(update={"debug": settings.debug.model_copy(update={"inject_tool": False})})
    assert "hands_inject" not in {t.name for t in Tracker(plain, Engine(plain), Stats(), View(20, 10)).tools}
