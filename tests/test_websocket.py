import asyncio
import json
import time

import pytest
import websockets

from neko.websocket import Signaler


@pytest.mark.asyncio
async def test_signaler_reconnects_and_routes_messages():
    url = "ws://localhost:8765"
    sig = Signaler(url)

    async def delayed_server():
        await asyncio.sleep(0.1)

        async def handler(ws):
            await ws.send(json.dumps({"event": "signal/offer", "sdp": "ok"}))
            await asyncio.sleep(1)

        async with websockets.serve(handler, "localhost", 8765):
            await asyncio.sleep(2)

    server_task = asyncio.create_task(delayed_server())

    start = time.monotonic()
    await asyncio.wait_for(sig.connect_with_backoff(), timeout=8)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.9

    msg = await asyncio.wait_for(sig.broker.topic_queue("control").get(), timeout=1)
    assert msg["event"] == "signal/offer"

    await sig.close()
    await server_task
