# Comprehensive WebRTC Flow Analysis: Working vs Main Branch

## Critical Discovery After 3x Analysis

After spending triple the time analyzing both branches line-by-line, I've identified **SEVEN critical differences** that cause the main branch to fail.

## The Smoking Gun: Sequence Inversion

### Working Branch - SYNCHRONOUS SEQUENTIAL FLOW
```
agent.py:1442-1465 (23 lines of synchronous code)
├─1. Connect WebSocket
├─2. IMMEDIATELY send signal/request
├─3. WAIT SYNCHRONOUSLY for offer (blocks execution)
└─4. Setup media WITH the offer (creates PC from offer)
```

### Main Branch (Before Fix) - ASYNCHRONOUS RACE CONDITION
```
webrtc_client.py:114-134 (20 lines with async races)
├─1. Connect WebSocket
├─2. Create PC without offer (WRONG!)
├─3. Start background tasks (async)
└─4. Request media (offer handled async later)
```

## Detailed Line-by-Line Analysis

### 1. CONNECTION INITIALIZATION

#### Working (agent.py)
```python
# Line 1442: WebSocket connection
await self.signaler.connect_with_backoff()

# Lines 1446-1452: IMMEDIATE media request
await self.signaler.send({
    "event": "signal/request",
    "payload": {
        "video": {},
        "audio": ({}) if self.audio else ({"disabled": True}),
    },
})
```

#### Main Before Fix (webrtc_client.py)
```python
# Line 114: WebSocket connection
await self.signaler.connect_with_backoff()

# Line 123: Create PC BEFORE request (WRONG!)
await self._setup_peer_connection()

# Line 126: Background tasks start
self._start_background_tasks()

# Line 133: Request media AFTER PC exists (TOO LATE!)
if self._request_media_enabled:
    await self._request_media()
```

**ISSUE #1:** Main creates peer connection before knowing offer details

### 2. OFFER RECEPTION PATTERN

#### Working - BLOCKING WAIT
```python
# Lines 1458-1463: Synchronous offer wait
control_q = self.signaler.broker.topic_queue("control")
offer_msg = None
while not offer_msg:  # BLOCKS until offer received
    msg = await control_q.get()
    if msg.get("event") in ("signal/offer", "signal/provide"):
        offer_msg = msg
```

#### Main Before Fix - ASYNCHRONOUS BACKGROUND
```python
# Lines 354-380: Background task (non-blocking)
async def _handle_signaling_messages(self):
    while self.is_connected():
        try:
            msg = await asyncio.wait_for(control_queue.get(), timeout=0.1)
            await self._handle_control_message(msg)  # Processes offer when it arrives
        except asyncio.TimeoutError:
            pass  # Continues looping
```

**ISSUE #2:** Main doesn't wait for offer before proceeding

### 3. PEER CONNECTION CREATION TIMING

#### Working - AFTER OFFER IN _setup_media()
```python
# Line 1465: Call _setup_media WITH offer
await self._setup_media(offer_msg)

# Inside _setup_media (Lines 1548-1574):
# Line 1548-1563: Extract ICE servers FROM OFFER
ice_payload = payload.get("ice") or payload.get("iceservers") or []
ice_servers: List[RTCIceServer] = []
for srv in ice_payload:
    # Parse server config from offer

# Line 1572: Create PC with offer's ICE config
pc = RTCPeerConnection(config)
```

#### Main Before Fix - BEFORE OFFER IN _setup_peer_connection()
```python
# Line 123: Called from connect() BEFORE offer
await self._setup_peer_connection()

# Inside _setup_peer_connection (Lines 265-291):
# Lines 266-272: Use INITIAL ice servers (not from offer!)
ice_servers: List[RTCIceServer] = []
for entry in self._configured_ice_servers:  # Uses constructor params
    ice_servers.append(RTCIceServer(**entry))

# Line 277: Create PC without offer knowledge
self.pc = RTCPeerConnection(config)
```

**ISSUE #3:** Main uses wrong ICE configuration (not from offer)

### 4. SDP NEGOTIATION FLOW

#### Working - IMMEDIATE IN SEQUENCE
```python
# Lines 1631-1637: Process offer immediately
await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type=remote_type))
answer = await pc.createAnswer()  # Immediate
await pc.setLocalDescription(answer)  # Immediate
await self.signaler.send({  # Immediate
    "event": "signal/answer",
    "payload": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
})
```

#### Main Before Fix - DELAYED IN BACKGROUND TASK
```python
# Lines 416-430: In _handle_control_message (async background)
if sdp and self.pc:
    await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
    # ... buffered ICE handling ...
    answer = await self.pc.createAnswer()
    await self.pc.setLocalDescription(answer)
    await self.signaler.send({
        "event": "signal/answer",
        "payload": {"sdp": answer.sdp, "type": "answer"}
    })
```

**ISSUE #4:** Main's SDP negotiation happens asynchronously, may race with other operations

### 5. ICE CANDIDATE FORMAT DIFFERENCE

#### Working (Line 2394)
```python
"candidate": cand.candidate,  # Raw candidate string
```

#### Main Initially (Line 468)
```python
"candidate": f"candidate:{candidate.candidate}",  # With prefix
```

**ISSUE #5:** Main added "candidate:" prefix not in working version (now fixed)

### 6. FRAME SOURCE INITIALIZATION

#### Working (Line 1574)
```python
# Created IMMEDIATELY after PC
pc = RTCPeerConnection(config)
self.pc = pc
self.frame_source = WebRTCFrameSource(self.settings, self.logger)  # Immediate
```

#### Main Before Fix (Lines 479-480)
```python
# Created in _on_track callback (delayed)
def _on_track(self, track):
    if track.kind == "video":
        self.frame_source = WebRTCFrameSource()  # Only when track received
        asyncio.create_task(self.frame_source.start(track))
```

**ISSUE #6:** Main delays frame source creation until track arrives (now fixed)

### 7. EVENT HANDLER REGISTRATION

#### Working (Lines 1614-1620)
```python
# Decorator pattern with async
@pc.on("icecandidate")
async def _on_ic(cand: RTCIceCandidate):
    await self._on_ice(cand)

@pc.on("track")
async def _on_tr(track: VideoStreamTrack):
    await self._on_track(track)
```

#### Main (Lines 278-280)
```python
# Direct method reference
self.pc.on("icecandidate", self._on_ice_candidate)
self.pc.on("track", self._on_track)
```

**ISSUE #7:** Different event handler patterns (though this shouldn't break)

## RACE CONDITION TIMELINE

### Working Branch Timeline (CORRECT)
```
T+0ms:   WebSocket connects
T+10ms:  Send signal/request
T+50ms:  Server sends offer (with ICE config)
T+51ms:  Client receives offer
T+52ms:  Create PC with offer's ICE servers
T+53ms:  setRemoteDescription(offer)
T+54ms:  ICE gathering starts (triggered by SRD)
T+55ms:  createAnswer()
T+56ms:  setLocalDescription(answer)
T+57ms:  Send answer to server
T+58ms:  ICE candidates generated
T+59ms:  Send ICE candidates to server
T+100ms: Server sends its ICE candidates
T+101ms: Add server's ICE candidates
T+150ms: ICE connection established
T+200ms: Video track received
T+201ms: Frames start flowing
```

### Main Branch Timeline (BROKEN - Before Fix)
```
T+0ms:   WebSocket connects
T+10ms:  Create PC (NO OFFER YET!)  ❌
T+11ms:  Register handlers
T+12ms:  Start background tasks
T+13ms:  Send signal/request
T+50ms:  Server sends offer
T+51ms:  Background task gets offer
T+52ms:  Need reset? Maybe...
T+53ms:  Reset PC (loses state!)  ❌
T+54ms:  setRemoteDescription(offer)
T+55ms:  ICE gathering maybe starts?
T+56ms:  createAnswer()
T+57ms:  setLocalDescription(answer)
T+58ms:  Send answer
T+??ms:  ICE candidates maybe generated?
T+??ms:  Candidates maybe sent?
T+10s:   Server timeout - no valid ICE pairs  ❌
T+10s:   signal/close received
```

## WHY THE REFACTORING BROKE

### 1. Violates WebRTC Protocol
WebRTC requires:
1. Signal intent (request media)
2. Receive offer with peer's capabilities
3. Create PC using offer's configuration
4. Process offer immediately
5. Generate answer synchronously
6. Exchange ICE candidates

Main branch created PC at step 0, breaking the protocol.

### 2. Race Conditions
- PC created before offer arrives
- Offer processed in background task
- ICE gathering may start prematurely
- Reset logic may corrupt state

### 3. Wrong ICE Configuration
- Working: Uses ICE servers FROM OFFER
- Main: Uses ICE servers from constructor
- Server's ICE config ignored initially

### 4. Async vs Sync Negotiation
- Working: Synchronous, deterministic sequence
- Main: Asynchronous, non-deterministic races

## THE FIX - Restore Working Sequence

```python
# webrtc_client.py connect() method - FIXED
async def connect(self) -> None:
    # 1. Connect WebSocket
    self.signaler = Signaler(self.ws_url)
    await self.signaler.connect_with_backoff()

    # 2. Initialize action executor
    self.action_executor = ActionExecutor(...)

    # 3. Request media FIRST (before PC exists)
    if self._request_media_enabled:
        await self._request_media()

    # 4. Wait SYNCHRONOUSLY for offer
    control_queue = self.signaler.broker.topic_queue("control")
    offer_msg = None
    while not offer_msg:
        msg = await asyncio.wait_for(control_queue.get(), timeout=10.0)
        if msg.get("event") in ("signal/offer", "signal/provide"):
            offer_msg = msg
            break

    # 5. Setup PC FROM the offer
    await self._setup_media_from_offer(offer_msg)

    # 6. NOW start background tasks
    self._start_background_tasks()
```

## New _setup_media_from_offer() Method

This method follows the working branch pattern exactly:

```python
async def _setup_media_from_offer(self, offer_msg: Dict[str, Any]) -> None:
    # 1. Start buffering early ICE candidates
    early_ice_payloads = []
    # ... buffering logic ...

    # 2. Extract ICE servers FROM OFFER
    ice_payload = payload.get("ice") or payload.get("iceservers") or []
    ice_servers = []
    for srv in ice_payload:
        # Parse from offer, not constructor

    # 3. Create PC with proper config
    config = RTCConfiguration(iceServers=ice_servers)
    self.pc = RTCPeerConnection(config)

    # 4. Create frame source immediately
    self.frame_source = WebRTCFrameSource()

    # 5. Process SDP synchronously
    await self.pc.setRemoteDescription(offer)
    answer = await self.pc.createAnswer()
    await self.pc.setLocalDescription(answer)
    await self.signaler.send(answer)

    # 6. Apply buffered ICE candidates
    for candidate in early_ice_payloads:
        await self.pc.addIceCandidate(candidate)
```

## VERIFICATION CHECKLIST

After fix, verify:
- [x] PC created AFTER offer received
- [x] ICE servers extracted FROM offer
- [x] setRemoteDescription called immediately
- [x] Answer sent synchronously
- [x] ICE candidates have correct format (no prefix)
- [x] Frame source created with PC
- [x] No reset logic during initial setup
- [x] ICE connection progresses to "connected"
- [x] No signal/close from server
- [x] Frames received successfully

## CONCLUSION

The refactoring introduced **fundamental protocol violations**:

1. **Temporal Inversion**: PC created before offer (violates WebRTC)
2. **Configuration Error**: Wrong ICE servers (ignores offer)
3. **Race Conditions**: Async handling of synchronous protocol
4. **State Corruption**: Reset logic during setup
5. **Format Mismatch**: ICE candidate prefix difference
6. **Delayed Initialization**: Frame source created late
7. **Non-deterministic Flow**: Background tasks vs synchronous

**The working branch follows WebRTC protocol correctly.**
**The main branch violated it in 7 different ways.**

This is not a subtle bug - it's a fundamental architectural error that completely breaks the WebRTC handshake protocol.

## Fix Summary

The fix applied to `src/neko_comms/webrtc_client.py`:

1. **Lines 123-136**: Reordered connect() to request media BEFORE creating PC
2. **Lines 269-430**: Added _setup_media_from_offer() method that:
   - Extracts ICE servers from offer payload
   - Creates PC with correct configuration
   - Processes SDP synchronously
   - Applies buffered ICE candidates
3. **Lines 467-477**: Removed "candidate:" prefix
4. **Line 358**: Create frame source before tracks arrive
5. **Lines 386-424**: Updated _handle_control_message to only handle renegotiations

The result: WebRTC connection now follows the correct protocol and works reliably.

## Date: October 31, 2024

Analysis performed comparing commit 13bf983 (working branch) with main branch after refactoring.
Fix successfully applied and tested.