import pytest

from neko.webrtc import build_configuration, create_peer_connection


@pytest.mark.asyncio
async def test_webrtc_offer_answer():
    config = build_configuration("stun:stun.example.org")
    pc1 = create_peer_connection(config)
    pc2 = create_peer_connection(config)
    pc1.createDataChannel("chat")

    try:
        offer = await pc1.createOffer()
        await pc1.setLocalDescription(offer)
        await pc2.setRemoteDescription(pc1.localDescription)
        answer = await pc2.createAnswer()
        await pc2.setLocalDescription(answer)
        await pc1.setRemoteDescription(pc2.localDescription)

        assert pc1.remoteDescription is not None
        assert pc2.remoteDescription is not None
    finally:
        await pc1.close()
        await pc2.close()
