# Neko
## Repository Cheat Sheet (Current Codebase)
- Project Structure:
  - `server/`: Go backend (`cmd/`, `internal/`, `pkg/`, optional `plugins/`). Build via `server/build` → `server/bin/neko`.
  - `client/`: Vue 2 + TypeScript SPA (`src/`, `public/`), built to `client/dist`.
  - `apps/`: Docker app images (e.g., `firefox/`, `chromium/`, `kde/`, `vlc/`).
  - `runtime/`: Base image and Xorg/driver configs; basis for final images.
  - `utils/`: Tooling; Dockerfile generator at `utils/docker/main.go`.
  - `webpage/`: Docusaurus docs site.
  - Root: `build` (image builder), `docker-compose.yaml` (local run), `config.yml` (sample config).
- Dev Commands:
  - Client dev: `cd client && npm ci && npm run serve` (hot-reload dev server).
  - Client build: `cd client && npm run build` → `client/dist`.
  - Client lint: `cd client && npm run lint`.
  - Server build: `cd server && ./build` (use `./build core` to skip plugins) → `server/bin/neko`.
  - Docker images: base `./build`, app `./build -a firefox`, flavor `./build -f nvidia -a chromium`.
  - Local run: `docker compose up -d` (serves `ghcr.io/m1k1o/neko/firefox:latest` on `:8080`).
- Coding Style:
  - Indent 2 spaces (LF). Client: Prettier single quotes, no semicolons, trailing commas. Go: gofmt/go vet; package names short/lowercase; files snake_case.
- Architecture & Entry Points:
  - SPA served as static assets from container; talks HTTP/WS on `:8080`.
  - `server/cmd/neko/main.go` → `cmd.Execute()`; `server/cmd/root.go` config/logging init; `server/cmd/serve.go` wires managers in order: session → member → desktop → capture → webrtc → websocket → api → http.
  - HTTP: `server/internal/http/manager.go` registers `/api`, `/api/ws`, `/health`, `/metrics`, static files, optional pprof.
  - REST router: `server/internal/api/router.go`. WebSocket: `server/internal/websocket/manager.go`. WebRTC: `server/internal/webrtc/manager.go`. Session/auth: `server/internal/session`, `server/internal/member/*`. Capture/Xorg: `server/internal/capture`, `server/internal/desktop`.
- API Surface (summary; see `server/openapi.yaml`):
  - Auth: `POST /api/login`, `POST /api/logout`, `GET /api/whoami`, `POST /api/profile`.
  - Sessions: `GET /api/sessions`, `GET/DELETE /api/sessions/{id}`, `POST /api/sessions/{id}/disconnect`.
  - Room: `GET/POST /api/room/settings`, broadcast `GET /api/room/broadcast`, `POST /api/room/broadcast/start|stop`.
  - Clipboard: `GET/POST /api/room/clipboard`, image `GET /api/room/clipboard/image.png`.
  - Keyboard: `GET/POST /api/room/keyboard/map`, `GET/POST /api/room/keyboard/modifiers`.
  - Control: `GET /api/room/control`, `POST /api/room/control/{request|release|take|give/{sessionId}|reset}`.
  - Screen: `GET/POST /api/room/screen`, `GET /api/room/screen/configurations`, `GET /api/room/screen/{cast.jpg|shot.jpg}`.
  - Upload: `POST /api/room/upload/{drop|dialog}`, `DELETE /api/room/upload/dialog`.
  - Utility: `POST /api/batch`, `GET /health`, `GET /metrics`.
- WebSocket:
  - Connect `ws(s)://<host>/api/ws` with cookie, `Authorization: Bearer <token>`, or `?token=<token>`.
  - Envelope `{ "event": "<string>", "payload": <JSON> }` with events defined under `server/pkg/types/event/events.go` (e.g., `system/init`, `signal/request`, `control/move`, `clipboard/set`, `keyboard/map`, `broadcast/status`, `file_chooser_dialog/opened|closed`).
- Configuration Model:
  - Defaults in `config.yml`; override via `NEKO_*` envs; legacy v2 envs supported behind `NEKO_LEGACY=1` and have v3 equivalents (see `server/internal/config/*`).

## Complete REST API (v3)
Authentication
- Methods: Cookie (`NEKO_SESSION`), Bearer token (`Authorization: Bearer <token>`), or query token (`?token=<token>`).
- Default security applies to all endpoints unless explicitly noted; `/health`, `/metrics`, and `/api/login` do not require auth.

General
- GET `/health`: Liveness probe. 200 text/plain.
- GET `/metrics`: Prometheus metrics. 200 text/plain.
- POST `/api/batch`: Execute multiple API requests in one call.
  - Request: `[{ path: string, method: 'GET'|'POST'|'DELETE', body?: any }]`.
  - Response: `[{ path, method, status: number, body?: any }]`.
- GET `/api/stats`: Server/session statistics.
  - Response: `{ has_host, host_id, server_started_at, total_users, last_user_left_at, total_admins, last_admin_left_at }`.

Current Session
- POST `/api/login`: Authenticate and start a session. No auth required.
  - Request: `{ username, password }`.
  - Response: `SessionLoginResponse` = `SessionData` plus optional `{ token }` if cookies are disabled.
- POST `/api/logout`: Terminate current session. 200.
- GET `/api/whoami`: Retrieve current session info.
  - Response: `SessionData`.
- POST `/api/profile`: Update current session’s runtime profile (no member sync).
  - Request: `MemberProfile`. 204.

Sessions
- GET `/api/sessions`: List active sessions.
  - Response: `SessionData[]`.
- GET `/api/sessions/{sessionId}`: Get a session by id.
  - Response: `SessionData`. 404 if not found.
- DELETE `/api/sessions/{sessionId}`: Remove a session. 204.
- POST `/api/sessions/{sessionId}/disconnect`: Force disconnect a session. 204.

Room Settings
- GET `/api/room/settings`: Get room settings.
  - Response: `Settings`.
- POST `/api/room/settings`: Update room settings. 204.
  - Request: `Settings`.

Room Broadcast
- GET `/api/room/broadcast`: Get broadcast status.
  - Response: `{ url?: string, is_active: boolean }`.
- POST `/api/room/broadcast/start`: Start RTMP broadcast.
  - Request: `{ url: string }`. 204. Errors: 400 missing URL; 422 already broadcasting; 500 start failure.
- POST `/api/room/broadcast/stop`: Stop broadcast. 204. Error: 422 not broadcasting.

Room Clipboard
- GET `/api/room/clipboard`: Get clipboard text/HTML.
  - Response: `{ text?: string, html?: string }`.
- POST `/api/room/clipboard`: Set clipboard text/HTML. 204.
  - Request: `{ text?: string, html?: string }`.
- GET `/api/room/clipboard/image.png`: Get clipboard image.
  - Response: `image/png` binary.

Room Keyboard
- GET `/api/room/keyboard/map`: Get keyboard map.
  - Response: `{ layout?: string, variant?: string }`.
- POST `/api/room/keyboard/map`: Set keyboard map. 204.
  - Request: `{ layout?: string, variant?: string }`.
- GET `/api/room/keyboard/modifiers`: Get keyboard modifiers.
  - Response: `{ shift?, capslock?, control?, alt?, numlock?, meta?, super?, altgr? }: boolean`.
- POST `/api/room/keyboard/modifiers`: Set modifiers. 204.
  - Request: same shape as response.

Room Control
- GET `/api/room/control`: Get control status.
  - Response: `{ has_host: boolean, host_id?: string }`.
- POST `/api/room/control/request`: Request host control. 204. Error: 422 already has host.
- POST `/api/room/control/release`: Release control. 204.
- POST `/api/room/control/take`: Take control (admin/host override). 204.
- POST `/api/room/control/give/{sessionId}`: Give control to session. 204.
  - Path: `sessionId` string. Errors: 400 target can’t host; 404 unknown session.
- POST `/api/room/control/reset`: Reset control state. 204.

Room Screen
- GET `/api/room/screen`: Get current screen config.
  - Response: `ScreenConfiguration`.
- POST `/api/room/screen`: Change screen config.
  - Request/Response: `ScreenConfiguration`. Errors: 422 invalid config.
- GET `/api/room/screen/configurations`: List available configurations.
  - Response: `ScreenConfiguration[]`.
- GET `/api/room/screen/cast.jpg`: Current screencast JPEG.
  - Response: `image/jpeg`. Errors: 400 screencast disabled; 500 fetch error.
- GET `/api/room/screen/shot.jpg`: On-demand screenshot JPEG.
  - Query: `quality` integer (0–100). Response: `image/jpeg`. Errors: 500 create image.

Room Upload
- POST `/api/room/upload/drop`: Upload files and drop at coordinates. 204.
  - multipart/form-data: `x: number`, `y: number`, `files: file[]`.
  - Errors: 400 upload error; 500 processing error.
- POST `/api/room/upload/dialog`: Upload files to active dialog. 204.
  - multipart/form-data: `files: file[]`. Errors: 422 no dialog.
- DELETE `/api/room/upload/dialog`: Close file chooser dialog. 204. Error: 422 no dialog.

Members
- GET `/api/members`: List members.
  - Query: `limit?: number`, `offset?: number`.
  - Response: `MemberData[]`.
- POST `/api/members`: Create member.
  - Request: `MemberCreate`.
  - Response: `MemberData`. Error: 422 ID exists.
- GET `/api/members/{memberId}`: Get member profile.
  - Response: `MemberProfile`. 404 if not found.
- POST `/api/members/{memberId}`: Update member profile. 204.
  - Request: `MemberProfile`.
- DELETE `/api/members/{memberId}`: Remove member. 204.
- POST `/api/members/{memberId}/password`: Update member password. 204.
  - Request: `{ password: string }`.
- POST `/api/members_bulk/update`: Bulk update member profiles. 204.
  - Request: `{ ids: string[], profile: MemberProfile }`.
- POST `/api/members_bulk/delete`: Bulk delete members. 204.
  - Request: `{ ids: string[] }`.

Schemas (Shapes)
- `SessionData`: `{ id: string, profile: MemberProfile, state: { is_connected: boolean, is_watching: boolean } }`.
- `MemberProfile`: `{ name?: string, is_admin?: boolean, can_login?: boolean, can_connect?: boolean, can_watch?: boolean, can_host?: boolean, can_share_media?: boolean, can_access_clipboard?: boolean, sends_inactive_cursor?: boolean, can_see_inactive_cursors?: boolean, plugins?: object }`.
- `MemberData`: `{ id: string, profile: MemberProfile }`.
- `MemberCreate`: `{ username: string, password: string, profile?: MemberProfile }`.
- `Settings`: `{ private_mode?, locked_controls?, implicit_hosting?, inactive_cursors?, merciful_reconnect?, plugins?: object }`.
- `ScreenConfiguration`: `{ width: number, height: number, rate: number }`.
- `KeyboardMap`: `{ layout?: string, variant?: string }`; `KeyboardModifiers`: all booleans listed above.
- `BroadcastStatus`: `{ url?: string, is_active: boolean }`.
- `ClipboardText`: `{ text?: string, html?: string }`.
- `ErrorMessage`: `{ message: string }` (for 4xx/5xx bodies).

Notes
- Unless stated, successful POSTs return 204 No Content; GETs return JSON bodies per schema.
- Image endpoints return binary JPEG/PNG with appropriate content types.
- Security schemes in effect globally: Bearer, Cookie, or token query; `/api/login` has security disabled.

## Complete WebSocket API
- Envelope: `{ "event": string, "payload": any }`. Connect to `ws(s)://<host>/api/ws` with cookie, Bearer, or `?token=`.
- Heartbeats: server sends `system/heartbeat` ~10s and low-level pings; clients may send `client/heartbeat` (no payloads).

System
- Server → Client `system/init`:
  - `{ session_id: string, control_host: { id?: string, has_host: boolean, host_id?: string }, screen_size: { width, height, rate }, sessions: { [id]: { id, profile: MemberProfile, state: SessionState } }, settings: Settings, touch_events: boolean, screencast_enabled: boolean, webrtc: { videos: string[] } }`.
- Server → Client (admins) `system/admin`:
  - `{ screen_sizes_list: ScreenSize[], broadcast_status: { is_active: boolean, url?: string } }`.
- Server → Client `system/settings`: `{ id: string, ...Settings }` (actor `id`, new settings).
- Client → Server `system/logs`: `[{ level: 'debug'|'info'|'warn'|'error', fields?: object, message: string }]`.
- Server → Client `system/disconnect`: `{ message: string }`.

Signaling (WebRTC)
- Client → Server `signal/request`: `{ video: { disabled?: boolean, selector?: { id: string, type: 'exact'|'best' }, auto?: boolean }, audio: { disabled?: boolean }, auto?: boolean }`.
- Server → Client `signal/provide`: `{ sdp: string, iceservers: [{ urls: string[], username?: string, credential?: string }], video: { disabled: boolean, id: string, video?: string, auto: boolean }, audio: { disabled: boolean } }`.
- Client ↔ Server `signal/offer` | `signal/answer`: `{ sdp: string }` (direction depends on negotiation; Neko often offers first via `signal/provide`).
- Client → Server `signal/candidate`: `{ candidate: string, sdpMid?: string, sdpMLineIndex?: number }`.
- Client → Server `signal/video`: `{ disabled?: boolean, selector?: { id: string, type: 'exact'|'best' }, auto?: boolean }`.
- Client → Server `signal/audio`: `{ disabled?: boolean }`.
- Server → Client `signal/restart`: `{ sdp: string }` (ICE restart offer).
- Server → Client `signal/close`: null payload (peer disconnected).

Sessions
- Server → Client `session/created`: `{ id, profile: MemberProfile, state: SessionState }`.
- Server → Client `session/deleted`: `{ id }`.
- Server → Client `session/profile`: `{ id, ...MemberProfile }`.
- Server → Client `session/state`: `{ id, is_connected, connected_since?, not_connected_since?, is_watching, watching_since?, not_watching_since? }`.
- Server → Client `session/cursors`: `[{ id: string, cursors: { x: number, y: number }[] }]` (only when `settings.inactive_cursors` enabled).

Control & Input
- Server ↔ Client `control/request`:
  - Client → Server: no payload (request host control).
  - Server → Client (to current host): `{ id: string }` (requester id).
- Server → Client `control/host`: `{ id: string, has_host: boolean, host_id?: string }`.
- Client → Server `control/release`: no payload (only host; requires `can_host`).
- Client → Server `control/move`: `{ x, y }`.
- Client → Server `control/scroll`: `{ delta_x?: number, delta_y?: number, control_key?: boolean }` (legacy `{ x, y }` supported).
- Client → Server `control/buttonpress|buttondown|buttonup`: `{ code: uint32, x?: number, y?: number }`.
- Client → Server `control/keypress|keydown|keyup`: `{ keysym: uint32, x?: number, y?: number }`.
- Client → Server `control/touchbegin|touchupdate|touchend`: `{ touch_id: uint32, x: number, y: number, pressure: uint8 }`.
- Client → Server `control/cut|control/copy|control/select_all`: no payload.
- Client → Server `control/paste`: `{ text?: string }` (optionally seeds clipboard first).
- Notes: Control requires `profile.can_host` and either being host or implicit hosting; clipboard ops also require `profile.can_access_clipboard`.

Screen
- Client → Server `screen/set`: `{ width, height, rate }` (admin only).
- Server → Client `screen/updated`: `{ id: string, width, height, rate }`.

Clipboard
- Client → Server `clipboard/set`: `{ text: string }` (host with clipboard permission).
- Server → Client `clipboard/updated`: `{ text: string }` (sent to host on OS clipboard change).

Keyboard
- Client → Server `keyboard/map`: `{ layout?: string, variant?: string }` (host only).
- Client → Server `keyboard/modifiers`: `{ shift?, capslock?, control?, alt?, numlock?, meta?, super?, altgr? }` (host only; omitted fields unchanged).

Broadcast
- Server → Client (admins) `broadcast/status`: `{ is_active: boolean, url?: string }`.

Opaque Send Channel
- Client → Server `send/unicast`: `{ receiver: string, subject: string, body: any }` → forwarded to receiver as `{ sender, receiver, subject, body }`.
- Client → Server `send/broadcast`: `{ subject: string, body: any }` → broadcast to others as `{ sender, subject, body }`.

File Chooser Dialog
- Server → Client `file_chooser_dialog/opened`: `{ id: string }` (host holding dialog).
- Server → Client `file_chooser_dialog/closed`: `{}`.

Shared Types (payload shapes)
- `MemberProfile`: `{ name?: string, is_admin?, can_login?, can_connect?, can_watch?, can_host?, can_share_media?, can_access_clipboard?, sends_inactive_cursor?, can_see_inactive_cursors?, plugins?: object }`.
- `SessionState`: `{ is_connected: boolean, connected_since?: string, not_connected_since?: string, is_watching: boolean, watching_since?: string, not_watching_since?: string }` (ISO timestamps).
- `Settings` (WS): `{ private_mode, locked_logins, locked_controls, control_protection, implicit_hosting, inactive_cursors, merciful_reconnect, heartbeat_interval, plugins?: object }`.
- `ScreenSize`: `{ width: number, height: number, rate: number }`.
- `KeyboardMap`: `{ layout: string, variant: string }`; `KeyboardModifiers`: booleans listed above.
- `Cursor`: `{ x: number, y: number }`.

### Heartbeat & Reconnect Behavior
- Layers of liveness
  - WebSocket ping/pong: Server sends a WS Ping every ~10s and also emits `system/heartbeat` in-band. Browsers auto‑reply with Pong; no client code is needed for WS Pong.
  - Client heartbeat event: Clients may send `client/heartbeat` at a cadence derived from `settings.heartbeat_interval` (default 120s). The server accepts it but does not strictly require it for liveness; it’s useful for analytics and legacy bridges.
- Settings and defaults
  - `session.heartbeat_interval` (default 120s) is included in `system/init.settings.heartbeat_interval` for the client to display/use.
  - `session.merciful_reconnect` (default true): If a session is “already connected” and a new WS arrives, the server replaces the old connection instead of rejecting it. With this off, a second connection is rejected with reason “already connected”.
- Timeouts and proxies
  - Ensure reverse proxies have read timeouts comfortably above both the WS Ping cadence (~10s) and the client heartbeat interval (≥120s). Recommended `proxy_read_timeout ≥ 300s` and to forward Upgrade/Connection headers for WebSocket.
  - Avoid aggressive idle timeouts on L4/L7 load balancers that terminate idle TCP flows under a few minutes; set ≥5 minutes where possible.
- Disconnect semantics
  - If WS Ping fails or the socket errors, the server closes the connection and sends `system/disconnect {message}` when possible. The session is marked disconnected; clients should auto‑reconnect and renegotiate WebRTC.
  - On WebRTC transport loss, the server emits `signal/close`; clients should re‑issue `signal/request` or handle `signal/restart`.
- Troubleshooting frequent drops
  - Blackouts every N seconds: increase reverse proxy `proxy_read_timeout`/keep‑alive timeouts; confirm no CDN/WebApp firewall is in the WS path.
  - “already connected” on reconnect: enable/keep `session.merciful_reconnect=true` (default), or ensure clients close prior tabs before reconnecting.
  - Mobile/background tabs: some OSes suspend WS/Pong; expect disconnects when backgrounded. The client must reconnect on resume.

### Legacy WebSocket Events (v2 Compatibility Proxy)
- Enabled when legacy mode is active (see `server/internal/http/manager.go` uses viper `legacy`). The legacy bridge maps v3 events to v2 event names for old clients and also emits some additional compatibility messages.
- Envelope: legacy messages also use `{event, ...payload}`.

- System:
  - `system/init`: `{ locks: {login?, control?, file_transfer?}: string map, implicit_hosting: boolean, file_transfer: boolean, heartbeat_interval: number }`.
  - `system/disconnect`: `{ title?: string, message: string }`.
  - `system/error`: historical; emitted by legacy paths on errors.
- Members:
  - `member/list`: `{ members: [{ id, name, admin: boolean, muted: boolean }] }`.
  - `member/connected`: `{ id, name, admin, muted }`.
  - `member/disconnected`: `{ id }`.
- Signaling:
  - `signal/provide`: `{ id: string, sdp: string, lite: boolean, ice: [{ urls, username?, credential? }] }`.
  - `signal/offer`: `{ sdp: string }`.
  - `signal/answer`: `{ displayname: string, sdp: string }`.
  - `signal/candidate`: `{ data: string }` (raw ICE JSON as string).
- Control/Admin (compatibility notifications):
  - `control/locked`: `{ id: string }` (lock established).
  - `control/release`: `{ id: string }`.
  - `control/request`: client-originated; `control/requesting`: notification to host; `control/give`: `{ id, target }`.
  - `control/clipboard`: `{ text: string }`.
  - `control/keyboard`: `{ layout?: string, capsLock?: boolean, numLock?: boolean, scrollLock?: boolean }`.
  - `admin/lock|unlock`: `{ resource: 'control'|'login'|'file_transfer', id: string }`.
  - `admin/control|release|give`: `{ id: string }` or `{ id, target }`.
  - `admin/ban|kick|mute|unmute`: `{ id: string }` or `{ target: string, id: string }`.
- Chat (legacy plugin messages):
  - `chat/message`: send/receive `{ id?: string, content: string }`.
  - `chat/emote`: send/receive `{ id?: string, emote: string }`.
- Filetransfer (legacy plugin messages):
  - `filetransfer/list`: `{ cwd: string, files: [{ name, size, is_dir, mtime, perms, ... }] }`.
  - `filetransfer/refresh`: same shape as list (triggered refresh).
- Screen/Broadcast:
  - `screen/configurations`: `{ configurations: { [idx]: { width, height, rates: { [idx]: rate } } } }`.
  - `screen/resolution`: `{ id?: string, width, height, rate }`.
  - `screen/set`: `{ width, height, rate }`.
  - `broadcast/status`: `{ url: string, isActive: boolean }`.
  - `broadcast/create`: `{ url: string }`; `broadcast/destroy`: no payload.

## 1. What Neko Is (Concept & Origin)
Neko (often styled **n.eko**) is an open‑source, self‑hosted *virtual* browser / remote desktop environment: you run a containerized Linux desktop with a preinstalled browser (Firefox, Chromium, etc.) on your own infrastructure; Neko streams the interactive desktop (video, audio, input) to remote clients via WebRTC, so multiple participants can watch and even take control in real time.  [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)

The project was started by its author after the shutdown of Rabb.it; needing a reliable way to watch anime remotely with friends over limited bandwidth + unstable Discord streaming, he built a WebRTC‑based Dockerized environment so everyone could share a single browser session. This collaborative genesis still shapes Neko’s multi‑user design (shared control queue, watch‑party friendliness).  [GitHub](https://github.com/m1k1o/neko) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)

Neko targets privacy, isolation, and portability: browsing happens in the container, not on the viewer’s device; host fingerprints/cookies stay server‑side; nothing persistent need touch the client unless you configure it. This “shielded browser” model is highlighted in both the docs and independent coverage (Heise), which also frames Neko as a lightweight VPN alternative for accessing internal resources without distributing full desktop access.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/)

## 2. Primary Use Cases
- **Collaborative browsing & watch parties:** All participants see the same live browser; host control can be passed; synchronized media playback works well because WebRTC streams the rendered video/audio from the container.  [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/)
- **Interactive presentations, workshops, remote support:** Presenter drives a shared browser/desktop; participants can be granted temporary control for demos or troubleshooting. Heise specifically calls out company trainings and support scenarios.  [GitHub](https://github.com/m1k1o/neko) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/)
- **Privacy / throwaway browsing / firewall bypass:** Because traffic originates from the Neko host, users can browse sites blocked locally (subject to policy/ethics); community reports note using Neko to get around locked‑down work networks.  [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [Reddit](https://www.reddit.com/r/selfhosted/comments/1ffz78l/neko_selfhosted_browser/)
- **Web dev & cross‑browser testing in controlled envs:** Spin up specific browser versions (incl. Waterfox, Tor, Chromium variants) to test sites without polluting local machines.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)
- **Remote application streaming beyond browsers:** Official images include full desktop environments (KDE, Xfce), Remmina (RDP/VNC client), VLC, and more; you can install arbitrary Linux GUI apps, turning Neko into a general remote app delivery layer.  [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)
- **Embedding into other web properties / programmatic rooms:** Docs and community guides show URL query param auth for frictionless embedding; REST API + Neko Rooms enable dynamic, ephemeral shareable sessions.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [GitHub](https://github.com/m1k1o/neko/releases) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/)
## 3. High‑Level Architecture
At a high level, a Neko deployment comprises:
- **Server container(s):** Run the Linux desktop + target browser/application; capture Xorg display frames + PulseAudio; encode via GStreamer; feed into WebRTC pipeline (Pion stack).  [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [GitHub](https://github.com/m1k1o/neko/releases)
- **Signaling / control plane:** HTTP + WebSocket endpoints manage sessions, auth, and host‑control; periodic ping/heartbeat maintain liveness (esp. behind proxies).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/releases)
- **WebRTC media plane:** ICE negotiation (STUN/TURN) to establish peer link(s); selectable port strategy (ephemeral range vs. UDP/TCP mux single port); optional Coturn relay for NAT‑restricted environments.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [GitHub](https://github.com/m1k1o/neko)
- **Client UI (served over HTTPS):** Browser front‑end page that renders the stream in a canvas/video element, sends input events (mouse/keyboard), displays participant cursors, chat/plugins, and exposes host‑control queue.  [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Optional ecosystem services:** REST API, Prometheus metrics exporter, plugin hooks (chat, file upload), and higher‑level orchestration projects (Neko Rooms / Apps / VPN).  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
## 4. Feature Inventory (v3 era)
- **Multi‑user concurrent session w/ host handoff + inactive cursors:** Participants can join; privileges (watch / host / share media / clipboard) governed per‑member profile.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
- **Audio + video streaming w/ low latency:** WebRTC transport from container to clients; GStreamer capture; stream selector to adjust quality.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
- **GPU acceleration modes (Intel/Nvidia flavors) & CPU builds:** Select appropriate image flavor to offload encoding & improve responsiveness; GPU support maturity varies—docs caution focus currently on CPU images.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
- **Granular auth/authorization (admin vs user; fine‑grained caps):** Role bits include can_login, can_connect, can_watch, can_host, can_share_media, can_access_clipboard, etc.; supports multiuser password split, file‑backed users, in‑memory object sets, and no‑auth (dev only).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/releases)
- **REST API + API token (admin programmatic control) & batch HTTP:** Added in v3; enables external orchestration, dynamic user provisioning, and admin operations without interactive login; API token should be short‑lived in ephemeral rooms.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
- **Prometheus metrics & pprof profiling:** Expose runtime health / performance metrics; integrate into observability stacks; profiling hooks assist tuning.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Desktop quality‑of‑life:** Clipboard reworked via xclip; drag‑and‑drop & file chooser upload; touchscreen input driver; dynamic resolution via xrandr; cursor image events.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Screencast endpoints + webcam/mic passthrough (experimental):** HTTP/JPEG screencast endpoints for snapshots/casting; optional upstream of user webcam/mic.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
- **Plugin system (chat, file upload, user‑scoped plugin config map).**  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
## 5. Supported Browsers / Apps / Desktops
Neko ships many tagged images; availability varies by architecture and GPU flavor. Current matrix (AMD64 strongest support): Firefox, Waterfox, Tor Browser; Chromium family incl. Google Chrome, Microsoft Edge, Brave, Vivaldi, Opera; plus Ungoogled Chromium. Additional desktop/media apps: KDE, Xfce, Remmina, VLC. ARM support exists for subsets (e.g., Brave & Vivaldi on ARM64; some lack DRM).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [GitHub](https://github.com/m1k1o/neko/releases)

Community packages (Umbrel) surface a streamlined install for home servers; Umbrel metadata shows current packaged version (3.0.4 at capture) and highlights collaboration + tunneling access patterns.  [apps.umbrel.com](https://apps.umbrel.com/app/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images)
## 6. Deployment Overview (Minimal to Advanced)
### 6.1 Quick Minimal Docker Run
Pull an image (e.g., Firefox flavor) and run mapping HTTP + WebRTC ports; provide screen size and user/admin passwords via env vars; share memory sized for modern browsers (e.g., 2GB). Community example docker‑compose (FOSS Engineer) shows mapping `8888:8080` plus `52000-52100/udp` EPR range and `NEKO_MEMBER_MULTIUSER_*` passwords.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
### 6.2 Choosing Registry & Tags
Prefer GitHub Container Registry (GHCR) for stable, flavor‑specific version tags; Docker Hub hosts latest dev (amd64) convenience builds. Semantic versioning (MAJOR.MINOR.PATCH) supported; `latest` for most recent stable—pin explicit tags for reproducibility.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [Docker Hub](https://hub.docker.com/r/m1k1o/neko)
### 6.3 Selecting Flavors (CPU vs GPU)
Image suffix selects hardware accel stack: `nvidia-*` for CUDA GPUs (AMD64), `intel-*` for VA‑API/QuickSync paths, or base CPU images. Docs caution GPU support may lag; verify in your environment.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)
### 6.4 Architecture Match & Resource Planning
Images published for linux/amd64, arm64, arm/v7; not every browser builds on all arches; some Chromium‑derived variants require ≥2GB RAM (Heise). Check the docs availability matrix before pulling.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)
### 6.5 Persistent State (Data Volumes)
While Neko can be run “throwaway,” you may bind‑mount config, member files, and persistent browser profiles to retain bookmarks, extensions (if policy permits), and user lists; docs show file/member providers referencing host paths (e.g., `/opt/neko/members.json`).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images)
## 7. Networking & WebRTC Ports
### 7.1 Why Ports Matter
WebRTC media does **not** traverse your HTTP reverse proxy; you must expose the negotiated media ports (or provide a TURN relay). If you only open 443 you will fail unless multiplexing or relay is used.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)
### 7.2 Ephemeral UDP Port Range (EPR)
Configure `NEKO_WEBRTC_EPR` (e.g., `59000-59100`) and expose identical host:container UDP range; don’t remap—ICE candidates must match reachable ports.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
### 7.3 UDP/TCP Multiplexing
Alternatively specify single `udpmux` / `tcpmux` ports when firewall pinholes are scarce; open both protocols for fallback where UDP blocked.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
### 7.4 Public vs NAT’d IPs
Set `nat1to1` when advertising a different reachable address (NAT hairpin caveats); or provide an IP retrieval URL to auto‑detect public address; otherwise ICE may hand out unroutable candidates.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
### 7.5 TURN Integration
Provide STUN/TURN server JSON (frontend/back‑end separation) via env vars; example Coturn compose snippet in docs; TURN recommended when clients sit behind strict NAT/firewalls.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
### 7.6 Real‑World Gotchas
Community reverse‑proxy thread shows mis‑set X‑Forwarded headers and missing additional port exposures leading to 502s; verifying correct WebRTC ports resolved issues for some users.  [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
## 8. Reverse Proxy Patterns (HTTP Plane)
### 8.1 Enable Proxy Trust
Set `server.proxy=true` so Neko honors `X-Forwarded-*` headers (important for logging, CSRF, cookie domain/path). Docs warn to adjust WebSocket timeouts because Neko pings every ~10s and expects client heartbeat ~120s.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
### 8.2 Traefik v2 Example
Label‑driven routing to backend `8080`; integrate TLS cert resolver; ensure UDP media ports separately exposed.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
### 8.3 Nginx Example & Header Hygiene
Minimal conf proxies HTTP + WebSocket upgrade; you may add X‑Forwarded‑For/Proto, cache bypass, and long read timeouts—legacy v2 docs show extended header set; community notes correcting `X-Forwarded-Proto` spelling vs “Protocol.”  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/reverse-proxy) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)
### 8.4 Apache, Caddy, HAProxy Templates
Docs provide working snippets incl. WebSocket rewrite for Apache; one‑liner `reverse_proxy` for Caddy w/ auto HTTPS; HAProxy ACL routing recipe w/ timeout tuning guidance.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/reverse-proxy)
## 9. Authentication & Authorization
### 9.1 Member vs Session Providers
Auth split: *Member Provider* validates credentials + returns capability profile; *Session Provider* persists session state (memory/file). Single member provider active at a time.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
### 9.2 Capability Flags (Granular Rights)
Per‑user profile booleans drive UI & backend enforcement: admin status; login/API; connect vs watch; host control; share media; clipboard access; send inactive cursor; see inactive cursors; plugin‑specific keys.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/releases)
### 9.3 Provider Types
- **Multiuser:** Two shared passwords (admin/user) generate ephemeral usernames; mirrors legacy v2 behavior.
- **File:** Persistent JSON map of users → hashed (optional) passwords + profiles.
- **Object:** In‑memory static list; no dup logins.
- **No‑Auth:** Open guest access (testing only—danger).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
### 9.4 API User Token
Separate non‑interactive admin identity for HTTP API calls; cannot join media; recommend ephemeral/rotated tokens (avoid long‑lived static in exposed rooms).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/releases)
### 9.5 Cookie Controls
Session cookie name, expiry, secure, httpOnly, domain/path configurable; disabling cookies falls back to token in client local storage—less secure (XSS risk). Keep cookies enabled for production.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
## 10. Security Considerations
-  **Surface reduction via containerization:** Browsing occurs inside an isolated container; you can discard state or run read‑only images for throwaway sessions; community privacy guides emphasize non‑retention setups.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [GitHub](https://github.com/m1k1o/neko)
- **Transport security & certs:** Terminate TLS at your reverse proxy (Traefik/Caddy/Certbot etc.); ensure WebSocket upgrades & long timeouts; see official reverse proxy examples.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/reverse-proxy)
- **Auth hardening:** Use strong unique admin/user passwords (or file/object providers w/ hashed credentials); avoid enabling no‑auth in public deployments; scope API tokens tightly.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Cookie vs token leakage:** Leaving cookies enabled (secure, httpOnly) prevents script access to session; disabling pushes token into JS‑accessible storage increasing exfiltration risk.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
- **Firewalling media ports:** Only expose required UDP/TCP ranges; where possible, restrict source IPs or require authenticated TURN; community reports of leaving ports closed manifest as connection failures rather than leaks—but mis‑config can open broad EPR ranges; plan network policy.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)
- **Extension install policy:** Browser policies in images may block arbitrary extension installs; you must explicitly allow if you need them—reduces attack surface by default.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images)
## 11. Performance & Tuning
- **Screen resolution & frame rate:** `NEKO_DESKTOP_SCREEN` / `NEKO_SCREEN` env controls virtual display mode (e.g., 1920x1080@30); higher rates = more bandwidth/CPU/GPU; choose based on clients & uplink.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Shared memory size:** Modern Chromium‑family browsers need large `/dev/shm`; examples allocate `shm_size: 2gb`; undersizing leads to crashes.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Bandwidth estimator (experimental adaptive bitrate):** Optional server‑side estimator can downgrade/upgrade encodes based on measured throughput; disabled by default; numerous thresholds/backoffs tunable.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [GitHub](https://github.com/m1k1o/neko/releases)
- **Hardware accel vs CPU encode tradeoffs:** GPU flavors reduce encode latency but add driver complexity; docs call out limited support maturity; Heise notes Neko can leverage Intel/Nvidia accelerated builds.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)
- **Resource guidance for Chromium variants:** Heise reports ≥2GB RAM allocation recommended when running Chromium‑based browsers in containers; plan host sizing accordingly.  [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images)
## 12. Administration & Operations
- **Logging & Debugging:** Enable debug logging via `log.level=debug` or env `NEKO_DEBUG=1`; GStreamer verbosity via `GST_DEBUG`; Pion debug by `PION_LOG_DEBUG=all`; inspect docker logs and browser dev console.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [GitHub](https://github.com/m1k1o/neko/releases)
- **Metrics & Profiling:** Prometheus metrics endpoint + pprof instrumentation introduced in v3 support operational monitoring and performance investigation.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Upgrades / Migration from v2:** Config modularization in v3; backward compatibility shims but deprecated; consult v3 docs + legacy reverse proxy header diffs when migrating.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/reverse-proxy) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)
- **Embedding Auto‑Login:** For kiosk/iframe use, append `?usr=<user>&pwd=<pwd>` to URL to bypass login prompt for viewers—use carefully; combine w/ restricted capability profile.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
- **Clipboard Behavior:** When accessed over HTTPS in supported host browsers (Chromium family), Neko hides its own clipboard button, deferring to native Clipboard API integration; not a bug.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [GitHub](https://github.com/m1k1o/neko/releases)
## 13. Ecosystem Projects
- **Neko Rooms:** Multi‑room orchestration wrapper that spins up independent Neko instances (ephemeral or persistent) with simplified onboarding (scripts, HTTPS via Let’s Encrypt, Traefik/NGINX automation); useful when you need per‑group isolation.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [Reddit](https://www.reddit.com/r/selfhosted/comments/1ffz78l/neko_selfhosted_browser/)
- **Neko Apps:** Library of containerized app bundles beyond browsers—expands use cases to general remote Linux app streaming; complements Rooms for scaling out multi‑app catalogs.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
- **Neko VPN (experimental):** Mentioned in docs nav as companion project enabling tunneled access paths; explore if you need integrated network overlay to reach internal apps through Neko.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)
- **Umbrel Packaging:** Curated home‑server integration; one‑click install, Umbrel tunneling for remote reachability, version tracking; good for homelab / non‑Docker‑experts.  [apps.umbrel.com](https://apps.umbrel.com/app/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
## 14. Comparison Touchpoints
- **vs. Kasm Workspaces:** Heise positions Neko as the lightweight alternative—Kasm provides full multi‑tenant workspace management & security layers but is heavier; Neko is simpler, container‑first, optimized for *shared* live sessions rather than individual isolated desktops (though you can run per‑user instances).  [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [GitHub](https://github.com/m1k1o/neko)
- **vs. Hyperbeam API (hosted embeddable co‑browse):** Neko offers a similar embeddable shared browser experience but is self‑hosted, giving you data control & on‑prem compliance; Heise explicitly calls out analogous embedding.  [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **vs. Generic Remote Desktop (VNC/NoVNC/Guacamole):** WebRTC yields smoother video + audio sync and lower interactive latency compared to image‑diff or poll‑based remotes; community commentary and docs emphasize superior streaming for media/watch usage.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [GitHub](https://github.com/m1k1o/neko)
## 15. Practical Config Snippets
```yaml
version: "3.4"
services:
neko:
  image: "ghcr.io/m1k1o/neko/firefox:latest"   # pick flavor/tag
  restart: unless-stopped
  shm_size: 2gb
  ports:
    - "8080:8080"                       # HTTP / signaling
    - "59000-59100:59000-59100/udp"     # WebRTC EPR
  environment:
    NEKO_DESKTOP_SCREEN: 1920x1080@30
    NEKO_MEMBER_MULTIUSER_ADMIN_PASSWORD: ${NEKO_ADMIN:?err}
    NEKO_MEMBER_MULTIUSER_USER_PASSWORD: ${NEKO_USER:?err}
    NEKO_WEBRTC_EPR: 59000-59100
    NEKO_WEBRTC_ICELITE: 1
    NEKO_DEBUG: 0
    # optionally front/back STUN/TURN JSON:
    # NEKO_WEBRTC_ICESERVERS_FRONTEND: '[{"urls":["stun:stun.l.google.com:19302"]}]'
    # NEKO_WEBRTC_ICESERVERS_BACKEND:  '[]'
volumes:
# mount for persistent member/session files if using file provider
# - ./data:/opt/neko
```
[fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)

```nginx
server {
listen 443 ssl http2;
server_name neko.example.com;

location / {
  proxy_pass http://127.0.0.1:8080;
  proxy_http_version 1.1;
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "upgrade";
  proxy_set_header Host $host;
  proxy_set_header X-Real-IP $remote_addr;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  proxy_set_header X-Forwarded-Proto $scheme;
  proxy_cache_bypass $http_upgrade;
}
}```
[neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/reverse-proxy) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)

```yaml
# Minimal member provider switch to file‑backed users with hashed passwords
member:
provider: file
file:
  path: "/opt/neko/members.json"
  hash: true
session:
file: "/opt/neko/sessions.json"
session:
api_token: "<short-lived-random-hex>"
```
[neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/releases)
## 16. Operational Runbook Checklist
- **Preflight:** Pick image flavor + arch; allocate ≥2GB RAM (Chromium); set shm_size; open media ports (EPR or mux); decide auth provider; create strong creds.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/installation/docker-images) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
- **Launch:** Compose up; confirm logs show listening on 8080 + WebRTC ports; test LAN client first; verify ICE candidates reachable (browser dev console).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
- **Secure:** Put behind TLS proxy; enable proxy trust; restrict ports/firewall; rotate API tokens; store hashed passwords.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
- **Scale / Multi‑tenant:** Use Neko Rooms or orchestration (k8s, compose bundles) to spin per‑team instances; leverage REST API + metrics for automation & autoscaling triggers.  [GitHub](https://github.com/m1k1o/neko/releases) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/)
- **Troubleshoot:** Turn on debug envs; inspect GStreamer logs for encode issues; validate reverse proxy headers; check that WebRTC ports aren’t blocked (common 502 confusion).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
## 17. Roadmap Glimpses & Future Directions
Recent release notes hint at additional session backends (Redis/Postgres), richer plugin ecosystem, and potential RDP/VNC relay modes where Neko acts as a WebRTC gateway rather than running the browser locally. Heise reports interest in direct protocol relay; docs flag “in the future” for expanded session providers.  [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/releases)
## 18. Community Lore / Field Notes
Homelabbers use Neko to co‑watch media, punch through restrictive corporate firewalls (when Neko host has outbound freedom), and expose full Linux desktops (KDE) to lightweight tablets. These anecdotes underscore why low‑latency WebRTC streaming and easy multi‑user control were prioritized.  [Reddit](https://www.reddit.com/r/selfhosted/comments/1ffz78l/neko_selfhosted_browser/) [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [GitHub](https://github.com/m1k1o/neko)

Reverse‑proxy misconfig (wrong header name, missing EPR exposure) is a recurring community stumbling block; always validate both HTTP and media planes.  [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
# Neko Browser + Playwright/CDP Integration Deep Dive
Neko (“n.eko”) is an open‑source, self‑hosted *virtual browser* that streams a full Linux desktop (not just a headless DOM) over WebRTC so multiple remote users can view and *interactively* control the same session in real time. It targets collaborative browsing, watch parties, remote support, embedded browser surfaces, and hardened “throwaway” cloud browsing where nothing persists locally.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction) [GitHub](https://github.com/m1k1o/neko) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html)

Unlike simple remote‑automation containers, Neko can run *any* Linux GUI application—browsers (Firefox, Chromium, Brave, Vivaldi, Waterfox, Tor, etc.), media players like VLC, full desktop environments (XFCE, KDE), and bespoke tools—because it captures an Xorg display and streams audio/video frames to clients via WebRTC. This breadth makes it viable for shared debugging sessions, interactive presentations, and as a privacy “jump box” into otherwise restricted networks.  [GitHub](https://github.com/m1k1o/neko) [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [Umbrel App Store](https://apps.umbrel.com/app/neko)

**Multi‑user collaboration** is first‑class: user roles, admin elevation, shared cursor visibility, host (keyboard/mouse) control arbitration, clipboard access, media sharing, and plugin‑scoped per‑user settings are governed by Neko’s v3 authentication system (Member + Session providers). This replaces v2’s simple dual‑password model and lets you express richer authorization matrices or plug in external identity sources.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/introduction)
### Versioning: v3 vs v2 & Legacy Mode
Neko v3 reorganized configuration into modular namespaces (server, member, session, webrtc, desktop, capture, plugins, etc.) and introduced providers; however, v3 retains *backward compatibility* with v2 environment variables when `NEKO_LEGACY=true` is set (and some legacy features auto‑detected). A migration table maps every major v2 var to its v3 equivalent (e.g., `NEKO_SCREEN`→`NEKO_DESKTOP_SCREEN`; `NEKO_PASSWORD`→`NEKO_MEMBER_MULTIUSER_USER_PASSWORD`; `NEKO_NAT1TO1`→`NEKO_WEBRTC_NAT1TO1`). This is critical when modernizing older compose files (like the snippet you shared) to avoid silent fallbacks and dual‑stream cursor quirks.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)

Heise’s Neko 3.0 coverage underscores why migrating matters: new browser flavors (Waterfox, additional Chromium builds, ARM variants), GPU‑accelerated options, HTTP/JPEG screencast endpoints, plugin ecosystem growth, and structural config changes—all shipping under a maintained Apache‑2.0 project—mean staying current pays dividends in stability and capability.  [heise online](https://www.heise.de/en/news/Virtual-browser-environment-Use-Firefox-Chrome-Co-in-Docker-with-Neko-3-0-10337659.html) [GitHub](https://github.com/m1k1o/neko)

Community quick‑start guides still widely circulate v2 envs (e.g., `NEKO_SCREEN`, `NEKO_PASSWORD`, `NEKO_ICELITE`, `NEKO_EPR`), which “work” only because legacy support remains—but they obscure v3 tuning knobs and can yield performance or auth surprises (e.g., no granular per‑user policy). Use the migration mapping to upgrade; I’ll show a patched compose below.  [fossengineer.com](https://fossengineer.com/selfhosting-neko-browser/) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2)
### Authentication Model (v3)
Authentication splits into **Member Provider** (who are you? what can you do?) and **Session Provider** (state & tokens). The *multiuser* provider emulates v2’s “user password” + “admin password” flow; you enable it via `NEKO_MEMBER_PROVIDER=multiuser`, then supply `NEKO_MEMBER_MULTIUSER_USER_PASSWORD` and `NEKO_MEMBER_MULTIUSER_ADMIN_PASSWORD`, optionally overriding default per‑role capability profiles (host, watch, clipboard, etc.). For tighter control, switch to *file* or *object* providers to define fixed accounts, hashed passwords, and granular profiles; or *noauth* for unsecured demo setups (never production). Session storage can persist to file; API access can be separately tokenized (`NEKO_SESSION_API_TOKEN`).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)

When exposing Neko programmatically (embedding in an app, auto‑provisioning rooms, LLM agents), consider disabling cookies or providing short‑lived API tokens; but weigh increased XSS risk if tokens leak into client JS when cookies are off. v3 exposes cookie flags (`secure`, `http_only`, domain/path scoping) so you can harden deployment behind TLS.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication)
### WebRTC Transport Essentials
For smooth low‑latency A/V + input streaming you *must* correctly expose Neko’s WebRTC ports. Three main patterns:

1. **Ephemeral UDP Port Range (EPR)** — Specify a contiguous range (e.g., `56000-56100`) via `NEKO_WEBRTC_EPR` and map the exact same range host:container *without remap*. Each new participant consumes ports; size range accordingly.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
2. **UDP/TCP Multiplexing** — Collapse to a single well‑known port (e.g., `59000`) as `NEKO_WEBRTC_UDPMUX` / `NEKO_WEBRTC_TCPMUX` for NAT‑challenged environments; trade throughput.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)
3. **ICE Servers** — Provide STUN/TURN front/back split: `NEKO_WEBRTC_ICESERVERS_FRONTEND` (what clients see) and `..._BACKEND` (what the server dials internally); JSON‑encoded arrays. Required when clients are off‑LAN and UDP paths are blocked.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)

If you run behind NAT, set `NEKO_WEBRTC_NAT1TO1` to the public (hairpin‑reachable) address; otherwise clients may ICE‑candidate a private IP and fail to connect. Automatic public IP fetch is available but you can override with `NEKO_WEBRTC_IP_RETRIEVAL_URL`.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2)

**Do not rely on your HTTP reverse proxy to relay WebRTC media.** Nginx/Traefik only front the signaling/control (HTTP(S)/WS) on port 8080; actual RTP/DTLS flows use the ports you expose above and must be reachable end‑to‑end or via TURN.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)

### Reverse Proxy & Timeouts
When fronting Neko with nginx/Traefik/etc., enable proxy trust in server config (`server.proxy=true` / `NEKO_SERVER_PROXY=1` in v3) so real client IPs from `X-Forwarded-*` are honored. Neko sends WS pings ~10s; clients heartbeat ~120s—so bump proxy read timeouts accordingly or users drop during long idle automation runs. Official nginx sample shows required `Upgrade`/`Connection` headers for WebSocket upgrade; community Nginx Proxy Manager threads confirm these plus extended `proxy_read_timeout` and forwarded IP headers to avoid 502s and broken control channels.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)
### Container Security / Chromium Sandboxing
Chromium inside containers often needs elevated namespaces to run its sandbox; many headless automation images either add `--no-sandbox` (reduced isolation) or grant `--cap-add=SYS_ADMIN` and supporting kernel flags so Chrome’s sandbox works. Puppeteer’s Docker docs call out the SYS_ADMIN requirement for their hardened image; Neko’s own v2 troubleshooting notes that forgetting SYS_ADMIN yields a black screen in Chromium variants—evidence the capability remains relevant. Decide: secure host kernel + allow SYS_ADMIN (preferred for full sandbox) *or* run `--no-sandbox` and accept risk; the sample supervisord snippet you posted already includes `--no-sandbox`, so SYS_ADMIN is belt‑and‑suspenders but still recommended for stability in GPU/namespace operations.  [pptr.dev](https://pptr.dev/guides/docker) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/troubleshooting)
### Enabling Chrome DevTools Protocol (CDP) in Neko for Playwright
Your goal: let humans drive the streamed Neko Chromium UI *and* attach automation via Playwright. Playwright supports attaching to any existing Chromium instance that exposes a DevTools endpoint via `chromium.connectOverCDP(endpointURL)`, where `endpointURL` can be the HTTP JSON version URL or direct WS endpoint; the returned `browser` exposes existing contexts/pages. Lower fidelity than full Playwright protocol, but ideal for “co‑drive” scenarios.  [Playwright](https://playwright.dev/docs/api/class-browsertype) [GitHub](https://github.com/m1k1o/neko/issues/391)

Once connected, you can open a raw **CDPSession** per page/context to send protocol commands (e.g., `Runtime.evaluate`, `Animation.enable`), mirroring the manual WebSocket probes in your `test.js`. This is useful for diagnostics, performance metrics, and low‑level tweaks Playwright doesn’t expose natively.  [Playwright](https://playwright.dev/docs/api/class-cdpsession) [Playwright](https://playwright.dev/docs/api/class-browsertype)
#### Remote Debugging Flags & Port Forward Pattern
Modern Chromium removed unrestricted `--remote-debugging-address=0.0.0.0` for security; recommended practice is bind the DevTools socket to localhost within the container (e.g., `--remote-debugging-port=9223`), then selectively forward or reverse‑proxy to an external port (e.g., 9222) with an auth / ACL layer (nginx, socat, SSH tunnel). Your nginx‑cdp sidecar implements precisely this 9222→9223 pass‑through with WebSocket upgrade and long timeouts—aligning with guidance from the Dockerized Chromium remote debugging discussion.  [Stack Overflow](https://stackoverflow.com/questions/58428213/how-to-access-remote-debugging-page-for-dockerized-chromium-launch-by-puppeteer) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)
### Review of Your `web-agent/neko-with-playwright` Compose Snippet
You posted a two‑service stack: `neko` (using `m1k1o/neko:chromium`) and an `nginx-cdp` sidecar in service network_mode sharing; supervisord launches Chromium with CDP flags and disables sandbox/gpu; nginx maps host 9222 to internal 9223 to front DevTools with WS keepalive/timeouts. Ports published: 52000→8080(tcp?) and 9222 (tcp). Issues & improvements:

- **1. Legacy Env Vars** – You’re mixing v2 (`NEKO_SCREEN`, `NEKO_PASSWORD*`, `NEKO_ICELITE`, `NEKO_NAT1TO1`) in a v3 world; while legacy support exists, you lose granular control and risk double cursor streams (cursor once in video, once separate) plus awkward auth extension later. Upgrade to v3 vars (`NEKO_DESKTOP_SCREEN`, `NEKO_MEMBER_PROVIDER=multiuser`, `NEKO_MEMBER_MULTIUSER_*`, `NEKO_WEBRTC_ICELITE`, `NEKO_WEBRTC_NAT1TO1`).  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)

- **2. Missing WebRTC Ports** – No UDP EPR or mux port is exposed, so remote WebRTC will fail off‑box unless clients are on the container host network and fallback mechanisms kick in. Add either an EPR range mapping and `NEKO_WEBRTC_EPR` or UDPMUX/TCPMUX single‑port mapping.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)

- **3. Public vs Private Subnet** – Your custom Docker subnet `17.100.0.0/16` collides with publicly routed Apple allocations (17.0.0.0/8 owned by Apple); choose RFC1918 (e.g., `172.31.0.0/16` or `10.67.0.0/16`) to avoid confusing clients seeing ICE candidates referencing real vs container ranges. Proper NAT1TO1 matters when advertising ICE addresses.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)

- **4. Proxy Headers & Timeouts** – Good start; ensure `proxy_read_timeout` is comfortably above Neko’s WebSocket ping interval (~10s)—set ≥60s or higher—and that `NEKO_SERVER_PROXY=1` (or config) is set so Neko trusts forwarded IPs; align with official reverse proxy doc + community NPM thread.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)

- **5. Chromium Capability / Sandbox** – You added `cap_add: SYS_ADMIN` (good) *and* `--no-sandbox` (less secure). Consider removing `--no-sandbox` once you confirm kernel support; Neko experiences black screens without SYS_ADMIN in Chromium images; Puppeteer’s hardened image docs reinforce giving SYS_ADMIN if you want sandbox.  [pptr.dev](https://pptr.dev/guides/docker) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/troubleshooting)

- **6. Password Hygiene** – Hard‑coding `neko` / `admin` is fine for testing but never production; switch to secrets or `.env` injection; multiuser provider makes it easy.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)

- **7. NAT Hairpin & ICE Lite** – You set `NEKO_ICELITE=0` (full ICE) and NAT1TO1 to container IP; if you actually need WAN access supply your public IP or domain; ICE Lite mode is only appropriate when server has public reflexive; official doc warns not to mix with external ICE servers.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc)

- **8. Debug Logging** – When diagnosing CDP or WebRTC handshake, enable `NEKO_DEBUG=1` and optional `GST_DEBUG` per FAQ; huge time saver.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
### Hardened & Modernized Compose Example (v3 Vars, CDP Enabled)
Below is an updated `docker-compose.yml` (org‑mode src). Key changes:
- Switched to GHCR explicit version tag (pin for reproducibility).
- RFC1918 subnet.
- Proper WebRTC EPR exposure.
- v3 auth vars.
- Proxy flag so Neko trusts sidecar.
- Optional API token for automation mgmt.
- Chromium started with localhost‑bound remote debugging; nginx sidecar terminates TLS (optional) & ACLs; you can env‑inject allowed upstream (e.g., ngrok tunnel).
- Dropped `--no-sandbox` (commented) to prefer secure sandbox; toggle per your threat model.
- Added healthcheck & log volumes.

```yaml
version: "3.8"

x-neko-env: &neko-env
NEKO_DESKTOP_SCREEN: 1920x1080@30
NEKO_MEMBER_PROVIDER: multiuser
NEKO_MEMBER_MULTIUSER_USER_PASSWORD: ${NEKO_USER_PASSWORD:-neko}
NEKO_MEMBER_MULTIUSER_ADMIN_PASSWORD: ${NEKO_ADMIN_PASSWORD:-admin}
NEKO_WEBRTC_EPR: 56000-56100          # match ports below
NEKO_WEBRTC_ICELITE: "false"          # full ICE unless static public IP
NEKO_WEBRTC_NAT1TO1: ${NEKO_PUBLIC_IP:-auto}  # set literal IP or leave unset to auto-detect
NEKO_SERVER_PROXY: "true"             # trust reverse proxy headers
NEKO_SESSION_API_TOKEN: ${NEKO_API_TOKEN:-}   # optional; blank disables
NEKO_DEBUG: ${NEKO_DEBUG:-0}

services:
neko:
  image: ghcr.io/m1k1o/neko/chromium:3.0.4
  container_name: neko
  restart: unless-stopped
  shm_size: 2gb
  networks:
    proxy:
      ipv4_address: 172.31.0.3
  ports:
    - "8080:8080/tcp"                 # web / signaling
    - "56000-56100:56000-56100/udp"   # WebRTC EPR (must match env)
  environment:
    <<: *neko-env
  cap_add:
    - SYS_ADMIN                       # required if not using --no-sandbox
  volumes:
    - neko-data:/var/lib/neko         # persistent config / sessions (bind as needed)
    - neko-logs:/var/log/neko
  configs:
    - source: supervisord_chromium
      target: /etc/neko/supervisord/chromium.conf

nginx-cdp:
  image: nginx:alpine
  container_name: neko-cdp
  network_mode: "service:neko"        # join same net & PID
  depends_on:
    - neko
  environment:
    # restrict which hosts may speak CDP (use allowlist or auth)
    ALLOWED_CDP_ORIGIN: ${ALLOWED_CDP_ORIGIN:-127.0.0.1}
  configs:
    - source: nginx_cdp_conf
      target: /etc/nginx/conf.d/cdp.conf
  ports:
    - "9222:9222/tcp"                 # exposed CDP endpoint proxied to 9223 in container

networks:
proxy:
  ipam:
    config:
      - subnet: 172.31.0.0/16

volumes:
neko-data:
neko-logs:

configs:
supervisord_chromium:
  content: |
    [program:chromium]
    environment=HOME="/home/%(ENV_USER)s",USER="%(ENV_USER)s",DISPLAY="%(ENV_DISPLAY)s"
    command=/usr/bin/chromium \
      --remote-debugging-port=9223 \
      --remote-debugging-address=127.0.0.1 \
      --remote-allow-origins="*" \
      --disable-web-security \
      --disable-features=VizDisplayCompositor \
      --disable-extensions \
      # --no-sandbox \  # uncomment only if you drop SYS_ADMIN
      --disable-dev-shm-usage \
      --enable-automation \
      --disable-background-timer-throttling \
      --disable-backgrounding-occluded-windows \
      --disable-renderer-backgrounding \
      --force-devtools-available \
      --disable-features=TranslateUI \
      --disable-ipc-flooding-protection \
      --enable-blink-features=IdleDetection \
      --headless=new \
      --disable-gpu
    stopsignal=INT
    autorestart=true
    priority=800
    user=%(ENV_USER)s
    stdout_logfile=/var/log/neko/chromium.log
    stdout_logfile_maxbytes=100MB
    stdout_logfile_backups=10
    redirect_stderr=true

    [program:openbox]
    environment=HOME="/home/%(ENV_USER)s",USER="%(ENV_USER)s",DISPLAY="%(ENV_DISPLAY)s"
    command=/usr/bin/openbox --config-file /etc/neko/openbox.xml
    autorestart=true
    priority=300
    user=%(ENV_USER)s
    stdout_logfile=/var/log/neko/openbox.log
    stdout_logfile_maxbytes=100MB
    stdout_logfile_backups=10
    redirect_stderr=true

nginx_cdp_conf:
  content: |
    map $http_upgrade $connection_upgrade {
      default upgrade;
      ''      close;
    }

    upstream chrome {
      server 127.0.0.1:9223;
      keepalive 32;
    }

    server {
      listen 9222;

      # Optional IP allowlist (simple example); extend w/ auth / mTLS as needed
      allow 127.0.0.1;
      allow ::1;
      # env-subst ALLOWED_CDP_ORIGIN could template additional allow lines
      deny all;

      location / {
        proxy_pass http://chrome;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_set_header Host $host:$server_port;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_connect_timeout 7200s;
        proxy_cache off;
        proxy_buffering off;
        proxy_max_temp_file_size 0;
        proxy_request_buffering off;

        proxy_set_header Sec-WebSocket-Extensions $http_sec_websocket_extensions;
        proxy_set_header Sec-WebSocket-Key $http_sec_websocket_key;
        proxy_set_header Sec-WebSocket-Version $http_sec_websocket_version;

        proxy_socket_keepalive on;
        keepalive_timeout 300s;
        keepalive_requests 1000;
      }
    }
```
[neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [Stack Overflow](https://stackoverflow.com/questions/58428213/how-to-access-remote-debugging-page-for-dockerized-chromium-launch-by-puppeteer) [pptr.dev](https://pptr.dev/guides/docker) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)
### Minimal `.env` Illustration (override at deploy)
```dotenv
NEKO_USER_PASSWORD=supersecretuser
NEKO_ADMIN_PASSWORD=supersecretadmin
NEKO_PUBLIC_IP=203.0.113.45        # example; or set DNS name in upstream LB/TURN
NEKO_API_TOKEN=$(openssl rand -hex 32)
NEKO_DEBUG=1
ALLOWED_CDP_ORIGIN=10.0.0.0/8      # example ACL range for automation runners
```
[neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)
### Playwright Attach Script (Improved)
Key best practices: discover the *browser* WebSocket endpoint from `/json/version`; create a context if none returned (some builds start w/ zero pages when headless new); gracefully handle targets; optionally filter to the Neko desktop window by URL. Example:

```js
// attach-neko.js
const { chromium } = require('playwright');

(async () => {
const cdpHttp = process.env.NEKO_CDP_URL || 'http://localhost:9222';

// Attach to existing Chromium exposed by Neko's CDP proxy.
const browser = await chromium.connectOverCDP(cdpHttp);

// In many cases Neko's running Chromium already has a default context.
// If none, create one.
const [defaultContext] = browser.contexts().length
  ? browser.contexts()
  : [await browser.newContext()];

// Reuse first existing page or open a new one.
const page = defaultContext.pages() || await defaultContext.newPage();

await page.goto('https://example.com');
console.log('Neko page title:', await page.title());

// Get raw CDP session if you want low-level control.
const client = await page.context().newCDPSession(page);
const version = await client.send('Browser.getVersion').catch(() => null);
console.log('CDP Browser Version:', version);

// Keep browser open for human co-driving; do NOT close().
})();
```
[Playwright](https://playwright.dev/docs/api/class-browsertype) [Playwright](https://playwright.dev/docs/api/class-cdpsession) [GitHub](https://github.com/m1k1o/neko/issues/391)
### Diagnostic CDP Ping Script (Refined from Your `test.js` / `test4.js`)
Below is a leaner diagnostic that:
1. Fetches `/json/version`;
2. Opens WebSocket;
3. Discovers targets;
4. Attaches to first non‑extension page;
5. Evaluates an expression;
6. Logs failures cleanly.

```js
// cdp-diagnostics.js
const WebSocket = require('ws');
const fetch = (...args) => import('node-fetch').then(({default: f}) => f(...args));

(async () => {
const base = process.env.NEKO_CDP_URL || 'http://localhost:9222';
const version = await (await fetch(`${base}/json/version`)).json();
const wsUrl = version.webSocketDebuggerUrl;

const ws = new WebSocket(wsUrl, { perMessageDeflate: false });
let id = 0;

function send(method, params, sessionId) {
ws.send(JSON.stringify({ id: ++id, method, params, sessionId }));
}

ws.on('open', () => {
console.log('CDP connected');
send('Target.setDiscoverTargets', { discover: true });
});

let firstSession;
ws.on('message', data => {
const msg = JSON.parse(data);
if (msg.method === 'Target.targetCreated') {
const t = msg.params.targetInfo;
if (t.type === 'page' && !t.url.startsWith('chrome-extension://')) {
send('Target.attachToTarget', { targetId: t.targetId, flatten: true });
}
} else if (msg.method === 'Target.attachedToTarget' && !firstSession) {
firstSession = msg.params.sessionId;
send('Runtime.enable', {}, firstSession);
send('Runtime.evaluate', { expression: '1+1' }, firstSession);
} else if (msg.id && msg.result) {
console.log('Result', msg.id, msg.result);
} else if (msg.error) {
console.error('CDP Error', msg.error);
}
});
})();
```
[Stack Overflow](https://stackoverflow.com/questions/58428213/how-to-access-remote-debugging-page-for-dockerized-chromium-launch-by-puppeteer) [Playwright](https://playwright.dev/docs/api/class-cdpsession) [Playwright](https://playwright.dev/docs/api/class-browsertype)
### Operational Checklist for Playwright‑Augmented Neko
| Check | Why | How to Verify |
| --- | --- | --- |
| Chromium started with `--remote-debugging-port` (localhost) | Required for CDP attach; safer than 0.0.0.0 | `curl http://<host>:9222/json/version` returns JSON |
| CDP proxy ACL in place | Prevent hostile takeover of your shared session | restrict IPs or auth in nginx; test from unauthorized host fails |
| WebRTC ports reachable | Avoid black screens / frozen video | `webrtc-internals` in client; `docker logs` ICE candidate errors |
| SYS_ADMIN vs `--no-sandbox` decision documented | Security posture clarity | Confirm container start flags; run `chrome://sandbox` |
| Multiuser passwords rotated | Prevent drive‑by admin | Use secrets; verify login roles mapping |
| Proxy timeout > heartbeat | Prevent surprise disconnects during long automation | Nginx `proxy_read_timeout >= 120s` |
| Debug logging toggled for incident response | Rapid triage | `NEKO_DEBUG=1`, `GST_DEBUG=3` when needed |
[neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [pptr.dev](https://pptr.dev/guides/docker) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/troubleshooting)
### Example Hybrid Workflow: Humans Steer, Agents Assist
A common pattern in agentic stacks:
1. Human opens Neko in browser, logs in as admin (multiuser).
2. Automation runner (Playwright script / LLM agent) attaches over CDP using service account limited by firewall.
3. Agent performs scripted setup (login, nav, cookie seeding) *then relinquishes*; human sees results instantly.
4. If human taking over triggers UI state changes, agent can poll via CDP events (Target/Runtime) to resume.

This model avoids re‑launching browsers and preserves session continuity Neko already streams to participants.  [Playwright](https://playwright.dev/docs/api/class-browsertype) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/authentication) [GitHub](https://github.com/m1k1o/neko/issues/391)
### Deployment Channels & Ecosystem
You can deploy via raw Docker/Compose, room orchestration stacks (neko‑rooms), homelab bundles (Umbrel App Store), or community charts/templates; packaging often pre‑wires reverse proxy + TLS but may lag in env var updates—review and update to v3 syntax after install.  [Umbrel App Store](https://apps.umbrel.com/app/neko) [GitHub](https://github.com/m1k1o/neko) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2)

### Troubleshooting Quick Hits
- **Black screen (cursor only) in Chromium flavor** → missing SYS_ADMIN or mis‑sandbox; confirm capability or drop sandbox flag.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v2/troubleshooting) [pptr.dev](https://pptr.dev/guides/docker)

- **WebRTC connect stalls / DTLS not started** → exposed UDP mismatch or firewall block; check EPR mapping & NAT1TO1; review server logs at debug level.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/configuration/webrtc) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/faq)

- **Users disconnect behind proxy** → heartbeat vs proxy timeout mismatch; ensure `proxy_read_timeout` >120s and `server.proxy` enabled.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup) [Reddit](https://www.reddit.com/r/nginxproxymanager/comments/ut8zyu/help_with_setting_up_reverse_proxy_custom_headers/)

- **CDP connect refused** → nginx sidecar not up or ACL blocking; verify `/json/version` at 9222 and upstream 9223 reachable in container.  [Stack Overflow](https://stackoverflow.com/questions/58428213/how-to-access-remote-debugging-page-for-dockerized-chromium-launch-by-puppeteer) [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/reverse-proxy-setup)

- **Legacy envs ignored** → upgrade to v3 names or set `NEKO_LEGACY=true` explicitly; review migration matrix.  [neko.m1k1o.net](https://neko.m1k1o.net/docs/v3/migration-from-v2)
# Neko v3 WebRTC & WebSocket Control: Frame/State, Keyboard, Mouse (Cited)
**TL;DR:**
- All browser control in Neko v3 is mediated over a single `/api/ws` WebSocket after session authentication.
- Browser frames are *not* delivered directly over the WS as video; rather, the WS carries *control*, *signaling*, *events*, and input (mouse/keyboard) JSON, with media frames (video, audio) negotiated via WebRTC ICE as a peer connection.
- Full workflow: REST login → WS upgrade (`/api/ws`) → system/init → WebRTC signal/request → ICE handshake → frames sent to client, controls sent from client.
## 1. Authenticate (REST, Cookie, Token, Password)
| Mode | REST Call | Response | WS Upgrade Auth |
| --- | --- | --- | --- |
| Cookie (default) | `POST /api/login {username, password}` | `Set-Cookie: NEKO_SESSION` | Cookie auto-sent |
| Token (stateless) | `POST /api/login` for `{token}` | Opaque JWT/Bearer | `?token=...` or Bearer header |
| Legacy (query) | (multiuser only) skip REST, `?password=` | — | ?password in query triggers v2 |
## 2. WebSocket Upgrade URL (With/Without Path Prefix)
- Mainline: `wss://host[:port]/api/ws?token=<TOKEN>`
- With path-prefix: e.g. `wss://proxy.example.com/neko/api/ws?token=...`
- Alt: cookies or `Authorization: Bearer ...` supported.
- Legacy `/ws` endpoint: only if enabled or in legacy mode.
## 3. Connection Lifecycle
1. **Upgrade:** Gorilla WS server handles `/api/ws`, performs token/cookie/Bearer/session check.
2. **Init:** Server pushes `system/init` (JSON: session id, settings, role).
3. **Heartbeat:** Server sends `system/heartbeat`; clients may reply with `client/heartbeat`, but liveness relies on WebSocket ping/pong.
4. **All interaction now flows over the socket:** control events (keyboard/mouse), signaling (ICE, SDP), system/broadcast, errors.
5. **All client state (host, cursor, input, session, etc.) is managed by events.**
## 4. How Media (Frames) and Control Flow
### Media
- **Video and audio frames** do **not** go over the WebSocket; they are WebRTC media streams (negotiated via signaling on WS).
- To *initiate* frame streaming, send:
  `{"event":"signal/request","payload":{"video":{},"audio":{}}}`
- Server replies with ICE candidates/SDP; client opens WebRTC peer connection.
- Browser’s actual frames (video, audio) arrive via WebRTC MediaStream.
- Some deployments **gate input on a completed WebRTC handshake**:
  1) send `{"event":"signal/request","payload":{"video":{},"audio":{}}}`
  2) perform SDP/ICE (offer/provide → answer; exchange candidates)
  3) only then will control events be honored.
### Input: Keyboard/Mouse
Input is sent from client to server as JSON events (current protocol):
- Pointer move: `{"event":"control/move","payload":{"x":123,"y":456}}`
- Mouse scroll: `{"event":"control/scroll", "payload": {"delta_x": 0, "delta_y": 120}}`
- Mouse press: `{"event":"control/buttonpress","payload":{"x":123,"y":456,"code":1}}`
- Mouse down: `{"event":"control/buttondown","payload":{"x":123,"y":456,"code":1}}`
- Mouse up: `{"event":"control/buttonup","payload":{"x":123,"y":456,"code":1}}`
	- button codes: 1=left, 2=middle, 3=right
- Key press: `{"event":"control/keypress","payload":{"keysym":65}}`
- Key down: `{"event":"control/keydown","payload":{"keysym":65}}`
- Key up: `{"event":"control/keyup","payload":{"keysym":65}}`
	- printable chars use ASCII; control keys use X11 keysyms (e.g., Enter=65293, Esc=65307, Arrows=65361–65364)

Host arbitration:
- Request: `{"event":"control/request"}`
- Release: `{"event":"control/release"}`
- Note: many servers will implicitly grant host on first valid control event, but explicit `control/request` is cleaner.

Coordinates expected by server are **pixels**. If your client uses normalized [0..1], convert and clamp before send.

**Remove legacy examples** using: `control/click`, `control/mouse`, `control/key`, `control/request_host` — these are not recognized by current servers.

### Control quick reference (current)
| Action                 | Event                       | Payload schema                                  |
|------------------------|-----------------------------|--------------------------------------------------|
| Move pointer           | `control/move`              | `{ "x": int, "y": int }`                         |
| Mouse scroll           | `control/scroll`            | `{ "delta_x": int, "delta_y": int }`             |
| Mouse button press     | `control/buttonpress`       | `{ "x": int, "y": int, "code": 1|2|3 }`          |
| Mouse button down/up   | `control/buttondown/up`     | `{ "x": int, "y": int, "code": 1|2|3 }`          |
| Key press              | `control/keypress`          | `{ "keysym": int }`                              |
| Key down/up            | `control/keydown/up`        | `{ "keysym": int }`                              |
| Request host           | `control/request`           | `{}`                                            |
| Release host           | `control/release`           | `{}`                                            |

**Common keysyms:** Enter=65293, Backspace=65288, Tab=65289, Escape=65307, Left=65361, Up=65362, Right=65363, Down=65364, Control=65507.
**Button codes:** 1=left, 2=middle, 3=right.

## 5. Production-Grade Client Implementation Notes
Building a robust programmatic client requires handling several subtle but critical aspects of the Neko protocol beyond basic event sending.

### 5.1 Connection Management & Reconnection
A production client must anticipate WebSocket disconnects. Simply reconnecting is insufficient. The correct pattern is a **connection manager** that, upon a detected drop:
1.  Completely tears down the old state: cancel all background async tasks (like loggers and signal consumers), and close the `RTCPeerConnection`. This prevents resource leaks and "zombie" tasks.
2.  Initiates a new connection attempt, often with exponential backoff to avoid spamming a downed server.
3.  Upon successful reconnection, rebuilds all necessary components: a new `Signaler`, new background tasks, and a new `RTCPeerConnection` for the subsequent WebRTC handshake.

### 5.2 Dynamic Keyboard Mapping
While a client can maintain a static table of common X11 keysyms, the most robust approach is to listen for the `keyboard/map` event from the server. This event provides a JSON object mapping key names to their precise keysym values for the server's specific environment.
- **Compatibility:** A client should handle both legacy (`{...}`) and current (`{"map": {...}}`) payload shapes for this event.
- **Strategy:** Maintain a default, built-in keysym map but dynamically update or extend it with the values received from the server at runtime. This ensures maximum compatibility across different keyboard layouts and server versions.

### 5.3 Liveness and Error Handling
- **Heartbeats:** The server periodically sends a `system/heartbeat` event and also issues WebSocket pings. Clients may optionally send `client/heartbeat`, but connection liveness primarily depends on WebSocket ping/pong; lack of `client/heartbeat` alone does not cause disconnection.
- **Error Events:** A client should listen for and surface all `error/...` events. These provide crucial feedback from the server if a sent command was malformed, rejected, or failed, which is essential for debugging.

### 5.4 Robust Input Handling
- **Host Control:** A robust client should monitor the `control/host` event. If `auto-host` functionality is desired, the client should track its own `session_id` (from `system/init`) and automatically send a `control/request` if it observes that the host has been dropped (`has_host: false`) or given to another session.
- **Sequential Double-Click:** To reliably simulate a double-click, a client should send two `control/buttonpress` (or down/up pairs) events sequentially, with a small delay (e.g., 50-100ms) in between. Using `async.gather` or sending them simultaneously can result in the server's input handler swallowing one of the events.

### 5.5 Defensive WebRTC Handshake
When parsing the `signal/provide` or `signal/offer` events, a client should be defensive about the `iceservers` array.
- **Strict Field Mapping:** Only parse and use the fields relevant to `aiortc` or the target WebRTC library (typically `urls`, `username`, and `credential`). Ignore any extra fields to avoid errors.
- **URL Flexibility:** Be prepared to handle both a single `url` key and a `urls` array.

## 6. References and Further Reading
- [Neko GitHub](https://github.com/m1k1o/neko)
- [Neko v3 Docs: Configuration](https://neko.m1k1o.net/docs/v3/configuration)
- [aiortc Documentation](https://aiortc.readthedocs.io/en/latest/)
- [Python Websockets Library](https://websockets.readthedocs.io/en/stable/)
- [WS Event Types (Source)](https://raw.githubusercontent.com/m1k1o/neko/master/server/pkg/types/websocket.go)
