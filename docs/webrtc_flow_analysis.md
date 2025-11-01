# WebRTC Flow Analysis: Working vs Main Branch

## Executive Summary

The refactored code on `main` branch has a **critical control flow inversion** in the WebRTC setup sequence that prevents ICE candidates from being exchanged properly, causing the connection to stall in "checking" state.

## Issue Symptoms

**Main Branch (Broken):**
- ICE connection stuck in "checking" state
- Connection state stuck in "connecting" state
- Server sends `signal/close` after 10 seconds (timeout)
- No video frames received
- Error: "No frame received within 20 seconds of video track"

## Root Cause: Connection Sequence Inversion

### WORKING Branch Flow (src/agent.py)

```
1. Connect WebSocket with backoff (line 1442)
   └─> await self.signaler.connect_with_backoff()

2. IMMEDIATELY request media (lines 1446-1452)
   └─> await self.signaler.send({
         "event": "signal/request",
         "payload": {"video": {}, "audio": {...}}
       })

3. Wait synchronously for offer (lines 1458-1463)
   └─> control_q = self.signaler.broker.topic_queue("control")
   └─> while not offer_msg:
         msg = await control_q.get()
         if msg.get("event") in ("signal/offer", "signal/provide"):
           offer_msg = msg

4. Setup media FROM offer (line 1465)
   └─> await self._setup_media(offer_msg)
       ├─> Create RTCPeerConnection (line 1572)
       ├─> Register event handlers (lines 1576-1620)
       ├─> setRemoteDescription(offer) (line 1631)
       ├─> createAnswer() (line 1632)
       ├─> setLocalDescription(answer) (line 1633)
       └─> Send answer to server (lines 1634-1637)

5. Start ICE buffering DURING setup (lines 1508-1524)
   └─> Buffer ICE candidates that arrive before SRD

6. Apply buffered ICE candidates AFTER SRD (lines 1655-1661)
   └─> for pay in early_ice_payloads:
         ice = self._parse_remote_candidate(pay)
         if ice:
           await pc.addIceCandidate(ice)
```

### MAIN Branch Flow (BEFORE FIX - src/neko_comms/webrtc_client.py)

```
1. Connect WebSocket (line 115)
   └─> await self.signaler.connect_with_backoff()

2. Setup peer connection WITHOUT offer (line 124)
   └─> await self._setup_peer_connection()
       ├─> Create RTCPeerConnection (line 278)
       ├─> Register event handlers (lines 279-282)
       └─> NO SDP exchange here!

3. Start background tasks (line 127)
   └─> self._start_background_tasks()
       └─> Starts _handle_signaling_messages() task

4. Request media (line 134)  ⚠️ TOO LATE!
   └─> if self._request_media_enabled:
         await self._request_media()

5. Background task handles offer asynchronously (lines 354-432)
   └─> _handle_signaling_messages()
       └─> _handle_control_message(msg)
           └─> If offer received:
               ├─> Reset peer connection (lines 413-414)
               ├─> setRemoteDescription(offer) (line 418)
               ├─> Apply buffered ICE (lines 421-423)
               ├─> createAnswer() (line 425)
               ├─> setLocalDescription(answer) (line 426)
               └─> Send answer (lines 428-431)
```

## Critical Differences

### 1. **Request Media Timing** ❌

**Working:** Request media BEFORE peer connection exists
```python
# Line 1446 - BEFORE _setup_media()
await self.signaler.send({
    "event": "signal/request",
    "payload": {"video": {}, "audio": {...}}
})
```

**Main (Before Fix):** Request media AFTER peer connection created
```python
# Line 134 - AFTER _setup_peer_connection()
if self._request_media_enabled:
    await self._request_media()
```

**Impact:** The server may send the offer before the client is ready to handle it properly.

### 2. **Synchronous vs Asynchronous Offer Handling** ❌

**Working:** Waits synchronously for offer, blocks until received
```python
# Lines 1458-1463
offer_msg = None
while not offer_msg:
    msg = await control_q.get()
    if msg.get("event") in ("signal/offer", "signal/provide"):
        offer_msg = msg
```

**Main (Before Fix):** Handles offer asynchronously in background task
```python
# Lines 354-432 - Background task processes offers when they arrive
async def _handle_signaling_messages(self):
    # ...processes offers asynchronously
```

**Impact:** Race condition - the peer connection may be created before or after the offer arrives.

### 3. **Peer Connection Creation Timing** ❌

**Working:** Creates PC inside `_setup_media()` AFTER receiving offer
```python
# Line 1572 - Inside _setup_media(), AFTER offer received
pc = RTCPeerConnection(config)
# Then immediately processes the offer
await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type=remote_type))
```

**Main (Before Fix):** Creates PC in `_setup_peer_connection()` BEFORE offer exists
```python
# Line 278 - Called from connect() BEFORE request_media()
self.pc = RTCPeerConnection(config)
# Offer processing happens later in background task
```

**Impact:** The peer connection is initialized without knowledge of the remote peer's capabilities.

### 4. **ICE Candidate Buffering** ⚠️

**Working:** Buffers ICE candidates that arrive before setRemoteDescription
```python
# Lines 1508-1524 - Early ICE buffering
early_ice_payloads: List[Dict[str, Any]] = []
async def _buffer_ice():
    while buffer_running:
        msg = await ice_q.get()
        if msg.get("event") == "signal/candidate":
            early_ice_payloads.append(msg.get("payload") or {})

# Lines 1655-1661 - Apply after SRD
for pay in early_ice_payloads:
    ice = self._parse_remote_candidate(pay)
    if ice:
        await pc.addIceCandidate(ice)
```

**Main:** Also buffers, but timing is different
```python
# Lines 454-458 - Buffering
if self._remote_description_set:
    await self.pc.addIceCandidate(candidate)
else:
    self._ice_candidates_buffer.append(candidate)

# Lines 421-423 - Application
for candidate in self._ice_candidates_buffer:
    await self.pc.addIceCandidate(candidate)
```

**Impact:** Should work the same, but depends on correct flag management.

### 5. **ICE Candidate Format** ⚠️

**Working:** Sends candidate without "candidate:" prefix
```python
# Line 2394 - From _on_ice callback
"candidate": cand.candidate,
```

**Main (Initial):** Sends candidate WITH "candidate:" prefix
```python
# Line 469 - From _on_ice_candidate callback
"candidate": f"candidate:{candidate.candidate}",
```

**Note:** The prefix was added in main based on previous debugging, but this may not be the issue. The fix removed this prefix.

## Correct Sequence (WebRTC Specification)

According to WebRTC specification and working implementation:

1. **Signal Intent:** Client sends `signal/request` to server
2. **Receive Offer:** Server responds with `signal/offer` containing SDP
3. **Create Peer Connection:** Client creates `RTCPeerConnection` with ICE config from offer
4. **Set Remote Description:** `pc.setRemoteDescription(offer)` triggers ICE gathering
5. **Create Answer:** `pc.createAnswer()` generates answer SDP
6. **Set Local Description:** `pc.setLocalDescription(answer)` commits the answer
7. **Exchange ICE Candidates:** Both peers exchange candidates via signaling
8. **ICE Connection:** Peers establish connectivity using exchanged candidates

## Why Main Branch Failed

The main branch violated this sequence by:

1. Creating peer connection BEFORE requesting media
2. Not waiting for the offer before initializing the peer connection
3. Handling the offer asynchronously, which causes timing issues

When the peer connection is created without an offer:
- The ICE gathering may start prematurely
- The peer connection doesn't have the remote peer's ICE configuration
- Candidates may be generated but not properly exchanged
- The connection gets stuck in "checking" state waiting for valid candidate pairs

## The Fix Applied

Restored the working sequence in the refactored code:

```python
# In connect() method:
1. await self.signaler.connect_with_backoff()

2. # Request media FIRST
   await self._request_media()

3. # Wait synchronously for offer
   control_queue = self.signaler.broker.topic_queue("control")
   offer_msg = None
   while not offer_msg:
       msg = await asyncio.wait_for(control_queue.get(), timeout=10.0)
       if msg.get("event") in ("signal/offer", "signal/provide"):
           offer_msg = msg
           break

4. # Setup peer connection FROM the offer
   await self._setup_media_from_offer(offer_msg)
```

Where `_setup_media_from_offer()` now:
1. Extracts ICE servers from offer payload
2. Creates RTCPeerConnection with proper configuration
3. Registers event handlers
4. Processes the SDP offer immediately
5. Generates and sends answer
6. Applies any buffered ICE candidates

This matches the working branch's proven approach.

## Additional Observations

### Event Handler Registration

**Working:** Uses decorator pattern
```python
@pc.on("icecandidate")
async def _on_ic(cand: RTCIceCandidate):
    await self._on_ice(cand)
```

**Main:** Uses direct assignment (should work the same)
```python
self.pc.on("icecandidate", self._on_ice_candidate)
```

### Frame Source Initialization

**Working:** Creates frame source immediately after PC
```python
# Line 1574 - RIGHT after creating RTCPeerConnection
self.frame_source = WebRTCFrameSource(self.settings, self.logger)
```

**Main (Fixed):** Now creates frame source before tracks arrive
```python
# In _setup_media_from_offer() - Created before tracks
self.frame_source = WebRTCFrameSource()
```

This ensures frame source is ready when track arrives.

## Files Modified

1. **src/neko_comms/webrtc_client.py**
   - Lines 103-147: `connect()` method - completely refactored
   - Lines 269-430: Added `_setup_media_from_offer()` method
   - Lines 467-477: Removed "candidate:" prefix
   - Lines 479-491: Updated `_on_track` to use pre-created frame source

## Testing Verification

After applying fixes, verify:
1. ✅ ICE connection progresses from "checking" → "connected"
2. ✅ Connection state progresses from "connecting" → "connected"
3. ✅ No `signal/close` received from server
4. ✅ First frame received within 20 seconds
5. ✅ Agent can execute navigation tasks successfully

## Conclusion

The refactoring introduced a **fundamental control flow inversion** by creating the peer connection before requesting media and receiving the offer. This violates the WebRTC signaling protocol and prevents proper ICE candidate exchange.

The fix restored the synchronous request → wait → setup sequence from the working branch, ensuring the peer connection is created with full knowledge of the remote peer's capabilities before SDP negotiation begins.

## Commit: October 31, 2024

Fixed in commit that modified `src/neko_comms/webrtc_client.py` to follow the correct WebRTC handshake sequence.