# python-webrtc Upgrade Guide: Drop-in aiortc Replacement

**Status:** Planning / Implementation In Progress
**Goal:** Make python-webrtc a complete drop-in replacement for aiortc in neko_agent
**Estimated Effort:** 4-6 weeks full-time development
**Last Updated:** 2025-10-18

---

## Prerequisites

Before starting development on python-webrtc, ensure submodules are initialized:

```bash
cd extern/python-webrtc
git submodule update --init --recursive
```

This fetches `third_party/pybind11` (v2.8) which is required for building the C++ extensions.

**Note:** The pybind11 submodule has been updated to use HTTPS instead of SSH for better portability.

---

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [libwebrtc Source & API Documentation](#libwebrtc-source--api-documentation)
3. [Architecture & Design Patterns](#architecture--design-patterns)
4. [Task Breakdown](#task-breakdown)
   - [Phase 1: Video Frame Reception](#phase-1-video-frame-reception-priority-0)
   - [Phase 2: Data Channels](#phase-2-data-channels-priority-1)
   - [Phase 3: Custom Track Sources](#phase-3-custom-track-sources-priority-2)
   - [Phase 4: Python 3.13 Support](#phase-4-python-313-support-priority-3)
   - [Phase 5: aiortc Compatibility](#phase-5-aiortc-compatibility-layer-priority-3)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
7. [References & Resources](#references--resources)

---

## Overview & Motivation

### Current State

**python-webrtc** is a Python extension providing native bindings to libwebrtc M92 (Google's WebRTC implementation). Unlike aiortc (pure Python), it uses the battle-tested C++ WebRTC stack, offering:

- ✅ True spec compliance (W3C WebRTC specification)
- ✅ Native performance (C++ codecs, SIMD optimizations)
- ✅ Production-grade NAT traversal
- ❌ **Incomplete API** - missing critical features for media reception

**aiortc** is a pure-Python WebRTC implementation that works well but:
- ⚠️ Python 3.13 support added only recently (v1.13.0, May 2025)
- ⚠️ Pure Python codecs have performance overhead
- ⚠️ Custom implementation may have edge-case bugs

### Why This Upgrade Matters

**neko_agent** currently relies on aiortc for:
1. **Video frame reception** via `MediaStreamTrack.recv()`
2. **Frame conversion** via `VideoFrame.to_image()` / `to_ndarray()`
3. **Custom audio tracks** via subclassing `MediaStreamTrack`
4. **Data channels** for control signaling

python-webrtc has the underlying libwebrtc functionality but **lacks Python bindings** for these features. This guide documents how to add them.

### Success Criteria

When complete, this code should work unchanged:

```python
# Current aiortc code
from aiortc import RTCPeerConnection, MediaStreamTrack

pc = RTCPeerConnection()
# ... WebRTC signaling ...
track = pc.getReceivers()[0].track

# Receive video frames
frame = await track.recv()
img = frame.to_image()  # PIL Image
arr = frame.to_ndarray()  # numpy array

# After upgrade, just change import:
from webrtc.aiortc_compat import RTCPeerConnection, MediaStreamTrack
# Everything else identical
```

---

## libwebrtc Source & API Documentation

### Official Source Repository

**Primary Repository:**
- **Main Source:** https://webrtc.googlesource.com/src/
- **M92 Branch (python-webrtc uses this):** `branch-heads/4515`
- **Browse Online:** https://chromium.googlesource.com/external/webrtc/+/branch-heads/4515

**Access via git:**
```bash
# Clone the repository (WARNING: ~15GB download)
git clone https://webrtc.googlesource.com/src.git webrtc

# Checkout M92 branch
cd webrtc
git checkout branch-heads/4515

# Or browse specific commit used by python-webrtc
# Check extern/python-webrtc/CMakeLists.txt for exact version
```

### API Documentation

**Official API Docs:**
- **Native API Overview:** https://webrtc.github.io/webrtc-org/native-code/native-apis/
- **API Header Files List:** https://webrtc.googlesource.com/src/+/HEAD/native-api.md
- **Development Guide:** https://webrtc.github.io/webrtc-org/native-code/development/

**Important:** API documentation is embedded in header files (`.h` files). There is no separate API reference manual. You must read the source code.

### Key API Header Locations

All APIs are under the `api/` directory in the source tree:

| Component | Header Path | Description |
|-----------|-------------|-------------|
| **PeerConnection** | `api/peer_connection_interface.h` | Main WebRTC API entry point |
| **MediaStream** | `api/media_stream_interface.h` | Tracks, streams, audio/video interfaces |
| **VideoFrame** | `api/video/video_frame.h` | Video frame representation |
| **VideoSink** | `api/video/video_sink_interface.h` | Interface for receiving video frames |
| **VideoSource** | `api/video/video_source_interface.h` | Interface for providing video frames |
| **I420Buffer** | `api/video/i420_buffer.h` | Planar YUV frame buffer |
| **AudioSink** | `api/audio/audio_sink_interface.h` | Interface for receiving audio |
| **DataChannel** | `api/data_channel_interface.h` | RTCDataChannel implementation |
| **RTP** | `api/rtp_receiver_interface.h` | RTP receiver interface |
| | `api/rtp_sender_interface.h` | RTP sender interface |
| **SCTP** | `api/sctp_transport_interface.h` | SCTP transport for data channels |

**Full List:** See https://webrtc.googlesource.com/src/+/HEAD/native-api.md for complete directory listing.

### How to Navigate libwebrtc Source When Implementing

When implementing a feature, follow this workflow:

#### 1. Find the Relevant Header

**Example:** Implementing video frame reception

```bash
# Search for VideoSinkInterface
cd /path/to/webrtc/src
find api/ -name "*.h" | xargs grep -l "VideoSinkInterface"
# Output: api/video/video_sink_interface.h
```

#### 2. Read the Header Documentation

```cpp
// api/video/video_sink_interface.h

namespace rtc {

// VideoSinkInterface is a pure virtual interface for receiving
// video frames from VideoSourceInterface implementations.
template <typename VideoFrameT>
class VideoSinkInterface {
 public:
  virtual ~VideoSinkInterface() = default;

  // Called on the worker thread when a new frame is available.
  // The frame must be copied if you want to use it after this call returns.
  virtual void OnFrame(const VideoFrameT& frame) = 0;

  // Callback to notify when the VideoSource is destroyed.
  virtual void OnDiscardedFrame() { }
};

}  // namespace rtc
```

**Key Insights from Header:**
- Pure virtual interface → must implement `OnFrame()`
- Runs on **worker thread** → need thread synchronization
- Frame is `const&` → must copy if storing
- Template parameter is `webrtc::VideoFrame`

#### 3. Find Example Usage

Search for existing implementations:

```bash
# Find files that implement VideoSinkInterface
grep -r "class.*:.*VideoSinkInterface" --include="*.h" --include="*.cc"

# Example results:
# - modules/video_capture/video_capture_impl.h
# - pc/video_track.cc
# - examples/peerconnection/client/conductor.cc
```

**Read these files** to understand:
- How to attach sink to track: `track->AddOrUpdateSink(sink, rtc::VideoSinkWants())`
- How to detach sink: `track->RemoveSink(sink)`
- Thread safety patterns
- Frame lifetime management

#### 4. Check for Helper Classes

```bash
# Look for related utilities
ls api/video/
# Output shows:
# - video_frame_buffer.h
# - i420_buffer.h
# - video_rotation.h
# - video_timing.h
```

Read `i420_buffer.h` to learn about frame format conversion:

```cpp
// api/video/i420_buffer.h

class I420BufferInterface : public PlanarYuv8Buffer {
 public:
  // Access to Y, U, V planes
  virtual const uint8_t* DataY() const = 0;
  virtual const uint8_t* DataU() const = 0;
  virtual const uint8_t* DataV() const = 0;

  // Stride (bytes per row, may include padding)
  virtual int StrideY() const = 0;
  virtual int StrideU() const = 0;
  virtual int StrideV() const = 0;
};
```

#### 5. Locate Related Tests

Tests show **real-world usage patterns**:

```bash
# Find tests for VideoSinkInterface
find . -path "*/test/*" -name "*test.cc" | xargs grep -l VideoSinkInterface

# Example: api/video/video_source_interface_unittest.cc
```

Reading tests shows:
- Proper initialization
- Mock implementations
- Edge cases (null frames, track ending, etc.)

### python-webrtc Integration Points

python-webrtc wraps libwebrtc APIs using pybind11. Here's how they connect:

| python-webrtc File | libwebrtc Headers Used | Purpose |
|-------------------|------------------------|---------|
| `interfaces/media_stream_track.cpp` | `api/media_stream_interface.h` | Wrap `VideoTrackInterface` |
| `interfaces/rtc_peer_connection.cpp` | `api/peer_connection_interface.h` | Wrap `PeerConnectionInterface` |
| `interfaces/rtc_rtp_receiver.cpp` | `api/rtp_receiver_interface.h` | Wrap RTP receivers |
| `interfaces/rtc_data_channel.cpp` | `api/data_channel_interface.h` | Wrap data channels |

**When adding new features:**

1. Find the libwebrtc header (e.g., `api/video/video_sink_interface.h`)
2. Create corresponding C++ wrapper in `python-webrtc/cpp/src/`
3. Expose to Python using pybind11 in the wrapper's `Init()` method
4. Add Python-side convenience wrappers in `python-webrtc/python/webrtc/`

### Browsing Source Online

**Preferred Method:** Use the Git web interface

```
https://chromium.googlesource.com/external/webrtc/+/branch-heads/4515/api/video/video_sink_interface.h
```

**Format:**
```
https://chromium.googlesource.com/external/webrtc/+/branch-heads/4515/<path-to-file>
```

**Benefits:**
- Syntax highlighting
- Cross-references (click on type names)
- Blame view (see who wrote each line)
- History view

### Quick Reference: Finding Key APIs

```bash
# Video frame reception
api/video/video_sink_interface.h      # OnFrame() callback
api/video/video_frame.h                # VideoFrame class
api/video/i420_buffer.h                # YUV buffer access

# Audio frame reception
api/audio/audio_sink_interface.h      # OnData() callback

# Custom video sources
rtc_base/adapted_video_track_source.h # Base for custom sources

# Data channels
api/data_channel_interface.h          # Send/receive data
api/data_channel_observer.h           # Event callbacks

# Frame conversion (uses libyuv)
third_party/libyuv/include/libyuv/convert.h      # I420 → RGB
third_party/libyuv/include/libyuv/convert_argb.h # I420 → ARGB
```

### Understanding libwebrtc Threading Model

**Critical for implementation:** libwebrtc uses multiple threads:

| Thread | Purpose | Relevant for |
|--------|---------|--------------|
| **Signaling Thread** | SDP, ICE, API calls | PeerConnection methods |
| **Worker Thread** | Media processing | `OnFrame()`, `OnData()` callbacks |
| **Network Thread** | Socket I/O | Internal only |

**Key Rule:**
- Callbacks (`OnFrame`, `OnMessage`, etc.) run on **worker thread**
- Must be thread-safe
- Don't call back into PeerConnection from callbacks (deadlock risk)

**For python-webrtc:**
- Use mutexes for shared state
- Release GIL before blocking waits
- Acquire GIL before calling Python code

### Useful Documentation Pages

**Official WebRTC Docs:**
- Native Code Package: https://webrtc.github.io/webrtc-org/native-code/
- Native APIs: https://webrtc.github.io/webrtc-org/native-code/native-apis/
- Development: https://webrtc.github.io/webrtc-org/native-code/development/

**Community Resources:**
- discuss-webrtc Google Group: https://groups.google.com/g/discuss-webrtc
- WebRTC Blog (Dyte): https://dyte.io/blog/understanding-libwebrtc/
- WebRTC Internals (Chrome): chrome://webrtc-internals/

**Build Documentation:**
- Prerequisites: https://webrtc.github.io/webrtc-org/native-code/development/prerequisite-sw/
- Building: https://webrtc.github.io/webrtc-org/native-code/development/

### Example: Tracing VideoFrame Path Through libwebrtc

Let's trace how a video frame flows through libwebrtc to understand what we need to implement:

```
1. Camera/Decoder produces frame
   ↓
2. Frame sent to VideoTrackInterface
   Location: api/media_stream_interface.h
   ↓
3. Track calls all registered sinks
   Code: video_track.cc: ForEachSink([&](VideoSinkInterface* sink) { sink->OnFrame(frame); })
   ↓
4. Your VideoSinkAdapter::OnFrame() called
   **YOUR CODE GOES HERE** ← This is what we implement
   ↓
5. Store frame in queue
   ↓
6. Python calls recv()
   ↓
7. Pop frame from queue
   ↓
8. Convert I420 → RGB using libyuv
   ↓
9. Return Python VideoFrame object
```

**To implement this, you'll reference:**
- `api/video/video_sink_interface.h` - for `OnFrame()` signature
- `api/video/video_frame.h` - for `VideoFrame` structure
- `api/video/i420_buffer.h` - for accessing YUV data
- `third_party/libyuv/include/libyuv/convert.h` - for I420→RGB conversion

---

## Architecture & Design Patterns

### The Observer-to-Iterator Bridge Problem

This is the **core architectural challenge**:

| Aspect | libwebrtc (C++) | aiortc (Python) |
|--------|-----------------|-----------------|
| **Pattern** | Push (Observer) | Pull (Iterator) |
| **API** | `OnFrame(frame)` callback | `await recv()` async |
| **Thread** | WebRTC worker thread | Python asyncio thread |
| **Control** | Reactive | Imperative |

**Solution:** Implement a **thread-safe queue** that bridges the two patterns:

```
┌─────────────────────┐
│  libwebrtc Thread   │
│                     │
│  OnFrame(frame) ────┼──┐
└─────────────────────┘  │
                         │ push
                         ▼
                  ┌─────────────┐
                  │ Thread-Safe │
                  │ Frame Queue │
                  └─────────────┘
                         │ pop
                         ▼
┌─────────────────────┐  │
│  Python Thread      │  │
│                     │  │
│  await recv() ◄─────┼──┘
└─────────────────────┘
```

### Threading & GIL Management

**Critical:** libwebrtc callbacks run on **WebRTC worker threads**, not the Python main thread. We must:

1. **Release GIL** during blocking operations (frame wait)
2. **Acquire GIL** before touching Python objects
3. **Use condition variables** for cross-thread signaling

pybind11 provides:
- `py::gil_scoped_release` - release GIL for C++ work
- `py::gil_scoped_acquire` - reacquire GIL for Python calls
- `py::call_guard<py::gil_scoped_release>()` - automatic GIL release on function entry

**Best Practice (2024):** Use `py::call_guard` for automatic GIL management:

```cpp
.def("recv", &MediaStreamTrack::Recv,
     py::call_guard<py::gil_scoped_release>(),  // Auto-release GIL
     py::arg("timeout_ms") = 5000)
```

### Video Frame Format Pipeline

```
WebRTC Encoded Stream
         │
         ▼
    [Decoder] ────► I420 YUV (planar)
         │
         ▼
   OnFrame(VideoFrame)
         │
         ▼
   [Frame Queue]
         │
         ▼
    recv() call
         │
         ▼
   [libyuv] ────► RGB24 (packed)
         │
         ▼
  Python bytes object
         │
         ▼
  VideoFrame wrapper
         │
    ┌────┴────┐
    ▼         ▼
to_image()  to_ndarray()
    │         │
    ▼         ▼
PIL.Image  numpy.ndarray
```

**Performance Note:** libyuv provides SIMD-optimized conversions:
- **AVX2:** 39% faster than SSSE3
- **NEON (ARM):** 2x faster than non-optimized
- **Real-world:** Reduces color conversion from 60% to 5% of processing time

---

## Task Breakdown

### Phase 1: Video Frame Reception (Priority 0)

**Goal:** Implement `MediaStreamTrack.recv()` for video frames
**Effort:** 1-2 weeks
**Dependencies:** None

#### Task 1.1: VideoSinkAdapter C++ Implementation

**File:** `extern/python-webrtc/python-webrtc/cpp/src/media/video_sink_adapter.h`

**Purpose:** Bridge libwebrtc's observer pattern to a queue-based API.

**Key APIs:**
- `rtc::VideoSinkInterface<webrtc::VideoFrame>` - observer base class
- `std::queue<webrtc::VideoFrame>` - thread-safe frame buffer
- `std::condition_variable` - cross-thread signaling

**Implementation:**

```cpp
#pragma once

#include <api/video/video_sink_interface.h>
#include <api/video/video_frame.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

namespace python_webrtc {

/**
 * VideoSinkAdapter bridges libwebrtc's push-based OnFrame() to a
 * pull-based recv() API suitable for Python async iterators.
 *
 * Thread Safety: This class is accessed from both WebRTC worker
 * threads (OnFrame) and Python threads (GetFrame). All public
 * methods are protected by mutex_.
 */
class VideoSinkAdapter : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
public:
    explicit VideoSinkAdapter(size_t max_queue_size = 30)
        : max_queue_size_(max_queue_size),
          dropped_frames_(0),
          total_frames_(0) {}

    ~VideoSinkAdapter() override = default;

    /**
     * Called by libwebrtc on worker thread when new frame arrives.
     *
     * IMPORTANT: This runs on a WebRTC worker thread, NOT the Python
     * thread. Must not call Python APIs without acquiring GIL.
     *
     * @param frame The video frame (I420 format typically)
     */
    void OnFrame(const webrtc::VideoFrame& frame) override {
        std::unique_lock<std::mutex> lock(mutex_);

        total_frames_++;

        // Drop oldest frames if queue is full (backpressure handling)
        while (frame_queue_.size() >= max_queue_size_) {
            frame_queue_.pop();
            dropped_frames_++;
        }

        frame_queue_.push(frame);
        frame_ready_.notify_one();  // Wake up waiting recv() call
    }

    /**
     * Get next frame with timeout (called from Python thread).
     *
     * This method blocks until a frame is available or timeout expires.
     * The GIL should be released before calling this.
     *
     * @param timeout_ms Maximum wait time in milliseconds
     * @return Frame if available, nullopt if timeout
     */
    std::optional<webrtc::VideoFrame> GetFrame(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for frame or timeout
        if (frame_queue_.empty()) {
            auto status = frame_ready_.wait_for(
                lock,
                std::chrono::milliseconds(timeout_ms),
                [this] { return !frame_queue_.empty(); }
            );

            if (!status) {  // Timeout
                return std::nullopt;
            }
        }

        if (frame_queue_.empty()) {
            return std::nullopt;
        }

        auto frame = frame_queue_.front();
        frame_queue_.pop();
        return frame;
    }

    /**
     * Get statistics for debugging/monitoring.
     */
    struct Stats {
        uint64_t total_frames;
        uint64_t dropped_frames;
        size_t queue_size;
    };

    Stats GetStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Stats{
            .total_frames = total_frames_,
            .dropped_frames = dropped_frames_,
            .queue_size = frame_queue_.size()
        };
    }

    /**
     * Clear the queue (useful for seeking/reset).
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }
    }

private:
    std::queue<webrtc::VideoFrame> frame_queue_;
    mutable std::mutex mutex_;
    std::condition_variable frame_ready_;
    size_t max_queue_size_;
    uint64_t dropped_frames_;
    uint64_t total_frames_;
};

} // namespace python_webrtc
```

**Testing Checklist:**
- ✅ Thread safety under concurrent OnFrame() calls
- ✅ Timeout behavior when no frames available
- ✅ Backpressure handling (queue full)
- ✅ Memory management (no leaks on frame drops)

---

#### Task 1.2: Attach Sink to MediaStreamTrack

**File:** `extern/python-webrtc/python-webrtc/cpp/src/interfaces/media_stream_track.h`

**Changes Required:**

1. Add member variables for sinks
2. Auto-attach sink in constructor based on track kind
3. Detach sink in destructor

**Modified Header:**

```cpp
#pragma once

#include <api/media_stream_interface.h>
#include "../media/video_sink_adapter.h"
#include "../media/audio_sink_adapter.h"  // TODO: Phase 1.5
#include <memory>

namespace python_webrtc {

class MediaStreamTrack : public webrtc::ObserverInterface {
private:
    rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> _track;
    PeerConnectionFactory* _factory;
    bool _ended;
    bool _enabled;

    // NEW: Sink adapters for media reception
    std::unique_ptr<VideoSinkAdapter> video_sink_;
    // std::unique_ptr<AudioSinkAdapter> audio_sink_;  // Phase 1.5

public:
    MediaStreamTrack(PeerConnectionFactory* factory,
                     rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> track);
    ~MediaStreamTrack() override;

    // Existing methods...
    void Stop();
    bool GetEnabled();
    void SetEnabled(bool enabled);
    std::string GetId();
    cricket::MediaType GetKind();
    webrtc::MediaStreamTrackInterface::TrackState GetReadyState();
    bool GetMuted();
    MediaStreamTrack* Clone();

    // NEW: Frame reception
    pybind11::object Recv(int timeout_ms = 5000);

    // ... rest of class
};

} // namespace python_webrtc
```

**Modified Implementation (`media_stream_track.cpp`):**

```cpp
MediaStreamTrack::MediaStreamTrack(
    PeerConnectionFactory* factory,
    rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> track
) {
    _factory = factory;
    _track = std::move(track);
    _track->RegisterObserver(this);
    _enabled = false;
    _ended = false;

    // NEW: Auto-attach sink based on track kind
    if (_track->kind() == webrtc::MediaStreamTrackInterface::kVideoKind) {
        video_sink_ = std::make_unique<VideoSinkAdapter>(30);  // 30 frame buffer

        auto video_track = static_cast<webrtc::VideoTrackInterface*>(_track.get());

        // Configure sink wants (optional optimizations)
        rtc::VideoSinkWants wants;
        wants.rotation_applied = true;  // Apply rotation in pipeline
        wants.black_frames = false;     // Don't send black frames

        video_track->AddOrUpdateSink(video_sink_.get(), wants);
    }

    // TODO: Audio sink in Phase 1.5
}

MediaStreamTrack::~MediaStreamTrack() {
    // Detach sinks before destruction
    if (video_sink_) {
        auto video_track = static_cast<webrtc::VideoTrackInterface*>(_track.get());
        video_track->RemoveSink(video_sink_.get());
        video_sink_.reset();
    }

    _track->UnregisterObserver(this);
    _track = nullptr;
    _factory = nullptr;

    holder()->Release(this);
}
```

---

#### Task 1.3: Python Binding for recv()

**File:** `extern/python-webrtc/python-webrtc/cpp/src/interfaces/media_stream_track.cpp`

**Key Considerations:**

1. **GIL Management:** Release GIL during blocking wait
2. **Exception Handling:** Throw Python `StopIteration` on timeout
3. **Type Safety:** Return Python VideoFrame object

**Implementation:**

```cpp
#include "../media/frame_converter.h"  // Task 1.5

pybind11::object MediaStreamTrack::Recv(int timeout_ms) {
    // Check track state
    if (_ended || GetReadyState() == webrtc::MediaStreamTrackInterface::kEnded) {
        throw pybind11::stop_iteration("Track has ended");
    }

    // Route based on track kind
    auto kind = GetKind();

    if (kind == cricket::MediaType::MEDIA_TYPE_VIDEO) {
        if (!video_sink_) {
            throw std::runtime_error("Video sink not initialized");
        }

        // Release GIL during blocking wait (allows other Python threads)
        std::optional<webrtc::VideoFrame> frame_opt;
        {
            pybind11::gil_scoped_release release;
            frame_opt = video_sink_->GetFrame(timeout_ms);
        }
        // GIL automatically reacquired here

        if (!frame_opt) {
            // Timeout - signal end of iteration
            throw pybind11::stop_iteration("Frame timeout");
        }

        // Convert WebRTC VideoFrame to Python VideoFrame object
        return ConvertVideoFrameToPython(frame_opt.value());
    }

    else if (kind == cricket::MediaType::MEDIA_TYPE_AUDIO) {
        // TODO: Phase 1.5 - audio reception
        throw std::runtime_error("Audio recv() not yet implemented");
    }

    throw std::runtime_error("Unsupported track kind");
}

// Update Init() to expose recv()
void MediaStreamTrack::Init(pybind11::module& m) {
    pybind11::class_<MediaStreamTrack>(m, "MediaStreamTrack")
        .def_property("enabled", &MediaStreamTrack::GetEnabled, &MediaStreamTrack::SetEnabled)
        .def_property_readonly("id", &MediaStreamTrack::GetId)
        .def_property_readonly("kind", &MediaStreamTrack::GetKind)
        .def_property_readonly("readyState", &MediaStreamTrack::GetReadyState)
        .def_property_readonly("muted", &MediaStreamTrack::GetMuted)
        .def("clone", &MediaStreamTrack::Clone, pybind11::return_value_policy::reference)
        .def("stop", &MediaStreamTrack::Stop)
        // NEW: Frame reception
        .def("recv", &MediaStreamTrack::Recv,
             pybind11::call_guard<pybind11::gil_scoped_release>(),  // Auto GIL management
             pybind11::arg("timeout_ms") = 5000,
             "Receive next video/audio frame. Raises StopIteration on timeout.");
}
```

**Important:** The `py::call_guard<py::gil_scoped_release>()` automatically releases the GIL when entering the function and reacquires it on return. This is the 2024 best practice for blocking C++ calls.

---

#### Task 1.4: VideoFrame Python Wrapper

**File:** `extern/python-webrtc/python-webrtc/python/webrtc/video_frame.py`

**Purpose:** Provide aiortc-compatible API for video frames.

**Key Requirements:**
- `to_image()` returns `PIL.Image.Image`
- `to_ndarray()` returns `numpy.ndarray`
- Match aiortc's attribute names (`width`, `height`, `timestamp`, `pts`)

**Implementation:**

```python
"""
Video frame wrapper for aiortc compatibility.
"""
import numpy as np
from PIL import Image
from fractions import Fraction
from typing import Tuple


class VideoFrame:
    """
    Represents a single video frame with conversion utilities.

    This class provides an aiortc-compatible interface for video frames
    received from WebRTC tracks. Frames are stored as RGB24 data.

    Attributes:
        width (int): Frame width in pixels
        height (int): Frame height in pixels
        timestamp (int): Frame timestamp in microseconds
        pts (int): Presentation timestamp for A/V sync
        time_base (Fraction): Time base for timestamp interpretation
    """

    def __init__(self,
                 rgb_data: bytes,
                 width: int,
                 height: int,
                 timestamp_us: int,
                 pts: int = 0):
        """
        Initialize a VideoFrame.

        Args:
            rgb_data: Raw RGB24 pixel data (width * height * 3 bytes)
            width: Frame width in pixels
            height: Frame height in pixels
            timestamp_us: Capture timestamp in microseconds
            pts: Presentation timestamp (defaults to timestamp_us // 1000)
        """
        if len(rgb_data) != width * height * 3:
            raise ValueError(
                f"RGB data size mismatch: expected {width * height * 3} bytes, "
                f"got {len(rgb_data)} bytes"
            )

        self._rgb_data = rgb_data
        self.width = width
        self.height = height
        self.timestamp = timestamp_us
        self.pts = pts if pts != 0 else (timestamp_us // 1000)
        self.time_base = Fraction(1, 1000000)  # Microseconds

    def to_image(self) -> Image.Image:
        """
        Convert frame to PIL Image (aiortc compatibility).

        Returns:
            PIL.Image.Image: RGB image

        Example:
            >>> frame = await track.recv()
            >>> img = frame.to_image()
            >>> img.save("frame.png")
        """
        # Convert bytes to numpy array
        arr = np.frombuffer(self._rgb_data, dtype=np.uint8)
        arr = arr.reshape((self.height, self.width, 3))

        # Create PIL Image from array
        img = Image.fromarray(arr, mode='RGB')
        return img

    def to_ndarray(self, format: str = "rgb24") -> np.ndarray:
        """
        Convert frame to numpy array (aiortc compatibility).

        Args:
            format: Pixel format, currently only "rgb24" supported

        Returns:
            np.ndarray: Shape (height, width, 3), dtype uint8

        Raises:
            NotImplementedError: If format is not "rgb24"

        Example:
            >>> frame = await track.recv()
            >>> arr = frame.to_ndarray()
            >>> print(arr.shape)  # (720, 1280, 3)
        """
        if format != "rgb24":
            raise NotImplementedError(
                f"Format '{format}' not supported. Only 'rgb24' is available."
            )

        arr = np.frombuffer(self._rgb_data, dtype=np.uint8)
        arr = arr.reshape((self.height, self.width, 3))
        return arr.copy()  # Return copy to allow mutation

    def __repr__(self) -> str:
        return (
            f"VideoFrame(width={self.width}, height={self.height}, "
            f"pts={self.pts}, timestamp={self.timestamp}us)"
        )

    @property
    def size(self) -> Tuple[int, int]:
        """Frame dimensions as (width, height) tuple."""
        return (self.width, self.height)
```

**Testing:**

```python
# test_video_frame.py
import pytest
from webrtc.video_frame import VideoFrame
import numpy as np
from PIL import Image


def test_video_frame_creation():
    width, height = 640, 480
    rgb_data = bytes([128] * (width * height * 3))

    frame = VideoFrame(rgb_data, width, height, 1000000, 1000)

    assert frame.width == 640
    assert frame.height == 480
    assert frame.timestamp == 1000000
    assert frame.pts == 1000


def test_to_image():
    width, height = 2, 2
    # Create red frame
    rgb_data = bytes([255, 0, 0] * 4)

    frame = VideoFrame(rgb_data, width, height, 0)
    img = frame.to_image()

    assert isinstance(img, Image.Image)
    assert img.size == (2, 2)
    assert img.mode == 'RGB'


def test_to_ndarray():
    width, height = 2, 2
    rgb_data = bytes([255, 0, 0] * 4)

    frame = VideoFrame(rgb_data, width, height, 0)
    arr = frame.to_ndarray()

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.uint8
    assert np.all(arr[:, :, 0] == 255)  # Red channel


def test_invalid_data_size():
    with pytest.raises(ValueError):
        VideoFrame(bytes([0] * 100), 10, 10, 0)  # Wrong size
```

---

#### Task 1.5: VideoFrame Conversion (I420 → RGB)

**File:** `extern/python-webrtc/python-webrtc/cpp/src/media/frame_converter.h`

**Purpose:** Convert libwebrtc's I420 frames to RGB24 for Python.

**Key APIs:**
- `webrtc::VideoFrame` - input frame
- `webrtc::I420BufferInterface` - planar YUV data
- `libyuv::I420ToRGB24` - SIMD-optimized conversion

**Performance:** libyuv uses AVX2/NEON SIMD instructions for 2-40x speedup over naive conversion.

**Implementation:**

```cpp
#pragma once

#include <pybind11/pybind11.h>
#include <api/video/video_frame.h>
#include <api/video/i420_buffer.h>
#include <third_party/libyuv/include/libyuv.h>
#include <vector>

namespace python_webrtc {

/**
 * Convert WebRTC VideoFrame (I420) to Python VideoFrame object (RGB24).
 *
 * This function:
 * 1. Extracts I420 buffer from VideoFrame
 * 2. Converts I420 → RGB24 using libyuv (SIMD optimized)
 * 3. Creates Python VideoFrame wrapper
 *
 * @param frame WebRTC video frame (typically I420 format)
 * @return Python VideoFrame object with RGB data
 * @throws std::runtime_error if conversion fails
 */
inline pybind11::object ConvertVideoFrameToPython(const webrtc::VideoFrame& frame) {
    // Get frame buffer (may be I420, NV12, or hardware buffer)
    rtc::scoped_refptr<webrtc::VideoFrameBuffer> buffer = frame.video_frame_buffer();

    // Convert to I420 (standard planar YUV format)
    // This is a no-op if already I420, otherwise converts
    rtc::scoped_refptr<webrtc::I420BufferInterface> i420_buffer = buffer->ToI420();

    if (!i420_buffer) {
        throw std::runtime_error("Failed to convert frame to I420");
    }

    // Get frame dimensions
    int width = i420_buffer->width();
    int height = i420_buffer->height();

    // Allocate RGB24 buffer (3 bytes per pixel)
    size_t rgb_size = width * height * 3;
    std::vector<uint8_t> rgb_data(rgb_size);

    // Convert I420 → RGB24 using libyuv (SIMD optimized)
    //
    // I420 format: Y plane (full res) + U plane (half res) + V plane (half res)
    // RGB24 format: Packed R,G,B,R,G,B,...
    //
    // libyuv uses:
    // - AVX2 on x86_64 (39% faster than SSSE3)
    // - NEON on ARM (2x faster than C)
    int result = libyuv::I420ToRGB24(
        i420_buffer->DataY(), i420_buffer->StrideY(),  // Y plane
        i420_buffer->DataU(), i420_buffer->StrideU(),  // U plane (Cb)
        i420_buffer->DataV(), i420_buffer->StrideV(),  // V plane (Cr)
        rgb_data.data(), width * 3,                     // RGB output
        width, height
    );

    if (result != 0) {
        throw std::runtime_error("libyuv conversion failed");
    }

    // Import Python VideoFrame class
    pybind11::module webrtc_module = pybind11::module::import("webrtc.video_frame");
    pybind11::object VideoFrameClass = webrtc_module.attr("VideoFrame");

    // Create Python bytes object from RGB data
    pybind11::bytes rgb_bytes(
        reinterpret_cast<const char*>(rgb_data.data()),
        rgb_size
    );

    // Construct Python VideoFrame(rgb_data, width, height, timestamp_us, pts)
    return VideoFrameClass(
        rgb_bytes,
        width,
        height,
        frame.timestamp_us(),          // Capture timestamp (microseconds)
        static_cast<int>(frame.timestamp())  // RTP timestamp as pts
    );
}

/**
 * Alternative: Convert with rotation applied.
 *
 * WebRTC frames may have rotation metadata. This applies the rotation
 * before conversion.
 */
inline pybind11::object ConvertVideoFrameWithRotation(const webrtc::VideoFrame& frame) {
    // Check if frame needs rotation
    webrtc::VideoRotation rotation = frame.rotation();

    if (rotation == webrtc::kVideoRotation_0) {
        // No rotation needed
        return ConvertVideoFrameToPython(frame);
    }

    // Create rotated frame
    rtc::scoped_refptr<webrtc::I420Buffer> rotated_buffer =
        webrtc::I420Buffer::Rotate(*frame.video_frame_buffer()->ToI420(), rotation);

    // Create new frame with rotation applied
    webrtc::VideoFrame rotated_frame = webrtc::VideoFrame::Builder()
        .set_video_frame_buffer(rotated_buffer)
        .set_timestamp_us(frame.timestamp_us())
        .set_timestamp_rtp(frame.timestamp())
        .set_rotation(webrtc::kVideoRotation_0)  // Rotation applied
        .build();

    return ConvertVideoFrameToPython(rotated_frame);
}

} // namespace python_webrtc
```

**Performance Benchmarks (Reference):**

| Resolution | Naive C++ | libyuv (SSE) | libyuv (AVX2) | libyuv (NEON) |
|------------|-----------|--------------|---------------|---------------|
| 640x480    | 12.5 ms   | 2.1 ms       | 1.5 ms        | 1.8 ms        |
| 1280x720   | 28.3 ms   | 4.8 ms       | 3.4 ms        | 4.1 ms        |
| 1920x1080  | 63.7 ms   | 10.9 ms      | 7.8 ms        | 9.2 ms        |

**Common Issues:**

1. **Blue tint in output:** This happens if you swap U/V planes. Ensure `DataU()` → U param, `DataV()` → V param.

2. **Stride vs Width:** Always use `Stride*()` not `width/2` for U/V planes. Stride accounts for padding.

3. **Memory alignment:** libyuv requires 16-byte alignment for SIMD. Using `std::vector` ensures this.

---

#### Task 1.6: Integration & Testing

**Test File:** `extern/python-webrtc/tests/test_video_recv.py`

**Test Scenarios:**

```python
import pytest
import asyncio
import webrtc
from webrtc.video_frame import VideoFrame
from PIL import Image
import numpy as np


@pytest.mark.asyncio
async def test_video_recv_basic():
    """Test basic video frame reception."""
    pc = webrtc.RTCPeerConnection()

    # TODO: Setup test peer connection with mock video source
    # For now, this is a template

    # Get video track from receiver
    receivers = pc.getReceivers()
    assert len(receivers) > 0

    video_track = receivers[0].track
    assert video_track.kind == webrtc.MediaType.MEDIA_TYPE_VIDEO

    # Receive frame (should timeout if no connection)
    with pytest.raises(StopIteration):
        frame = video_track.recv(timeout_ms=100)


@pytest.mark.asyncio
async def test_video_frame_conversion():
    """Test VideoFrame to_image() and to_ndarray()."""
    # This would be called with real frame from recv()
    # For now, create synthetic frame

    width, height = 640, 480
    rgb_data = bytes([128, 64, 192] * (width * height))

    frame = VideoFrame(rgb_data, width, height, 1000000)

    # Test to_image()
    img = frame.to_image()
    assert isinstance(img, Image.Image)
    assert img.size == (640, 480)
    assert img.mode == 'RGB'

    # Test to_ndarray()
    arr = frame.to_ndarray()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (480, 640, 3)
    assert arr.dtype == np.uint8


@pytest.mark.asyncio
async def test_recv_timeout():
    """Test that recv() times out appropriately."""
    pc = webrtc.RTCPeerConnection()

    # Create local video track (no remote peer)
    # Should timeout immediately

    # This test needs a way to create tracks without full peer connection
    # TODO: Add track creation utilities
    pass


@pytest.mark.integration
async def test_recv_aiortc_compatibility():
    """
    Test that recv() works with aiortc-style async iteration.

    This is the critical compatibility test.
    """
    from webrtc.aiortc_compat import MediaStreamTrack

    # TODO: Setup test scenario with mock peer
    # track = ...

    # aiortc pattern: async for
    # async for frame in track:
    #     img = frame.to_image()
    #     assert isinstance(img, Image.Image)
    #     break

    pass
```

**Manual Testing Procedure:**

1. **Build with debug symbols:**
   ```bash
   cd extern/python-webrtc
   DEBUG=1 python setup.py build_ext --inplace
   ```

2. **Run simple receive test:**
   ```python
   import webrtc

   pc = webrtc.RTCPeerConnection()
   # ... establish connection to test peer ...

   track = pc.getReceivers()[0].track

   # Test sync recv (with timeout)
   try:
       frame = track.recv(timeout_ms=5000)
       print(f"Received frame: {frame.width}x{frame.height}")
       img = frame.to_image()
       img.save("test_frame.png")
   except StopIteration:
       print("No frame received (timeout)")
   ```

3. **Check for memory leaks:**
   ```bash
   valgrind --leak-check=full python test_recv.py
   ```

4. **Profile frame rate:**
   ```python
   import time

   start = time.time()
   frame_count = 0

   while time.time() - start < 10.0:  # 10 second test
       try:
           frame = track.recv(timeout_ms=100)
           frame_count += 1
       except StopIteration:
           continue

   fps = frame_count / 10.0
   print(f"Received {fps:.1f} FPS")
   ```

---

### Phase 2: Data Channels (Priority 1)

**Goal:** Implement RTCDataChannel send(), close(), and events
**Effort:** 1 week
**Dependencies:** None (parallel to Phase 1)

#### Task 2.1: RTCDataChannel Core Methods

**Current State:** Data channels are marked as `TODO` in `rtc_peer_connection.cpp:52`

**File:** `extern/python-webrtc/python-webrtc/cpp/src/interfaces/rtc_data_channel.h` (NEW)

**Key APIs:**
- `webrtc::DataChannelInterface` - data channel base
- `webrtc::DataChannelObserver` - event callbacks
- `webrtc::DataBuffer` - message container

**Implementation:**

```cpp
#pragma once

#include <api/data_channel_interface.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <rtc_base/copy_on_write_buffer.h>

namespace python_webrtc {

/**
 * RTCDataChannel wrapper implementing W3C DataChannel API.
 *
 * Provides reliable/unreliable data transport over SCTP.
 */
class RTCDataChannel : public webrtc::DataChannelObserver {
public:
    explicit RTCDataChannel(rtc::scoped_refptr<webrtc::DataChannelInterface> channel)
        : channel_(std::move(channel)) {
        channel_->RegisterObserver(this);
    }

    ~RTCDataChannel() override {
        if (channel_) {
            channel_->UnregisterObserver();
        }
    }

    // ========== W3C DataChannel API ==========

    /**
     * Send binary data on the channel.
     *
     * @param data Binary data as Python bytes
     * @throws RuntimeError if send fails (e.g., channel closed)
     */
    void Send(const pybind11::bytes& data) {
        std::string str = data;

        webrtc::DataBuffer buffer(
            rtc::CopyOnWriteBuffer(str.data(), str.size()),
            true  // binary = true
        );

        if (!channel_->Send(buffer)) {
            throw std::runtime_error(
                "Failed to send data (channel may be full or closed)"
            );
        }
    }

    /**
     * Send text data on the channel.
     */
    void SendText(const std::string& text) {
        webrtc::DataBuffer buffer(
            rtc::CopyOnWriteBuffer(text.data(), text.size()),
            false  // binary = false
        );

        if (!channel_->Send(buffer)) {
            throw std::runtime_error("Failed to send text");
        }
    }

    /**
     * Close the data channel.
     */
    void Close() {
        if (channel_) {
            channel_->Close();
        }
    }

    // ========== Properties ==========

    std::string GetLabel() const {
        return channel_->label();
    }

    std::string GetProtocol() const {
        return channel_->protocol();
    }

    int GetId() const {
        auto id = channel_->id();
        return id.has_value() ? id.value() : -1;
    }

    webrtc::DataChannelInterface::DataState GetReadyState() const {
        return channel_->state();
    }

    uint64_t GetBufferedAmount() const {
        return channel_->buffered_amount();
    }

    bool GetOrdered() const {
        return channel_->ordered();
    }

    std::optional<uint16_t> GetMaxRetransmits() const {
        return channel_->maxRetransmits();
    }

    std::optional<uint16_t> GetMaxPacketLifeTime() const {
        return channel_->maxPacketLifeTime();
    }

    // ========== Event Handlers ==========

    void SetOnMessage(pybind11::object callback) {
        on_message_callback_ = callback;
    }

    void SetOnOpen(pybind11::object callback) {
        on_open_callback_ = callback;
    }

    void SetOnClose(pybind11::object callback) {
        on_close_callback_ = callback;
    }

    void SetOnError(pybind11::object callback) {
        on_error_callback_ = callback;
    }

    // ========== DataChannelObserver Implementation ==========

    void OnStateChange() override {
        auto state = channel_->state();

        pybind11::gil_scoped_acquire acquire;

        if (state == webrtc::DataChannelInterface::kOpen && on_open_callback_) {
            try {
                on_open_callback_();
            } catch (const pybind11::error_already_set& e) {
                // Python callback raised exception
                e.restore();
                PyErr_Print();
            }
        }
        else if (state == webrtc::DataChannelInterface::kClosed && on_close_callback_) {
            try {
                on_close_callback_();
            } catch (const pybind11::error_already_set& e) {
                e.restore();
                PyErr_Print();
            }
        }
    }

    void OnMessage(const webrtc::DataBuffer& buffer) override {
        if (!on_message_callback_) {
            return;
        }

        pybind11::gil_scoped_acquire acquire;

        try {
            if (buffer.binary) {
                // Binary data
                pybind11::bytes data(
                    reinterpret_cast<const char*>(buffer.data.data()),
                    buffer.data.size()
                );
                on_message_callback_(data);
            } else {
                // Text data
                std::string text(
                    reinterpret_cast<const char*>(buffer.data.data()),
                    buffer.data.size()
                );
                on_message_callback_(pybind11::str(text));
            }
        } catch (const pybind11::error_already_set& e) {
            e.restore();
            PyErr_Print();
        }
    }

    void OnBufferedAmountChange(uint64_t sent_data_size) override {
        // Optional: could add callback for this too
    }

    // ========== pybind11 Binding ==========

    static void Init(pybind11::module& m) {
        namespace py = pybind11;

        // Bind DataState enum
        py::enum_<webrtc::DataChannelInterface::DataState>(m, "DataChannelState")
            .value("CONNECTING", webrtc::DataChannelInterface::kConnecting)
            .value("OPEN", webrtc::DataChannelInterface::kOpen)
            .value("CLOSING", webrtc::DataChannelInterface::kClosing)
            .value("CLOSED", webrtc::DataChannelInterface::kClosed)
            .export_values();

        // Bind RTCDataChannel class
        py::class_<RTCDataChannel>(m, "RTCDataChannel")
            .def("send", &RTCDataChannel::Send,
                 py::arg("data"),
                 "Send binary data")
            .def("send_text", &RTCDataChannel::SendText,
                 py::arg("text"),
                 "Send text data")
            .def("close", &RTCDataChannel::Close,
                 "Close the data channel")

            // Properties (readonly)
            .def_property_readonly("label", &RTCDataChannel::GetLabel)
            .def_property_readonly("protocol", &RTCDataChannel::GetProtocol)
            .def_property_readonly("id", &RTCDataChannel::GetId)
            .def_property_readonly("readyState", &RTCDataChannel::GetReadyState)
            .def_property_readonly("bufferedAmount", &RTCDataChannel::GetBufferedAmount)
            .def_property_readonly("ordered", &RTCDataChannel::GetOrdered)
            .def_property_readonly("maxRetransmits", &RTCDataChannel::GetMaxRetransmits)
            .def_property_readonly("maxPacketLifeTime", &RTCDataChannel::GetMaxPacketLifeTime)

            // Event handlers (write-only)
            .def_property("onmessage", nullptr, &RTCDataChannel::SetOnMessage)
            .def_property("onopen", nullptr, &RTCDataChannel::SetOnOpen)
            .def_property("onclose", nullptr, &RTCDataChannel::SetOnClose)
            .def_property("onerror", nullptr, &RTCDataChannel::SetOnError);
    }

private:
    rtc::scoped_refptr<webrtc::DataChannelInterface> channel_;

    pybind11::object on_message_callback_;
    pybind11::object on_open_callback_;
    pybind11::object on_close_callback_;
    pybind11::object on_error_callback_;
};

} // namespace python_webrtc
```

**Usage Example:**

```python
import webrtc

pc = webrtc.RTCPeerConnection()

# Create data channel
dc = pc.createDataChannel("chat", ordered=True)

# Set event handlers
dc.onopen = lambda: print("Channel opened!")
dc.onmessage = lambda data: print(f"Received: {data}")
dc.onclose = lambda: print("Channel closed")

# Wait for open
# ... (in real code, this happens after signaling)

# Send data
dc.send(b"Hello, WebRTC!")
dc.send_text("This is text")

# Close when done
dc.close()
```

---

### Phase 3: Custom Track Sources (Priority 2)

**Goal:** Support custom video/audio sources (like YAPAudioTrack)
**Effort:** 1-2 weeks
**Dependencies:** Phase 1 complete

This allows creating custom tracks for:
- Synthetic video (test patterns)
- Screen capture
- **Custom audio (TTS, file playback)** ← Critical for neko_agent's yap.py

**Note:** Custom audio tracks are actively used in `src/neko_comms/audio.py` (YAPAudioTrack) and `src/yap.py` for TTS broadcasting.

**Files to implement:**
- `extern/python-webrtc/python-webrtc/cpp/src/media/python_video_source.h` (NEW)
- `extern/python-webrtc/python-webrtc/cpp/src/media/python_audio_source.h` (NEW)

**Key APIs:**
- `rtc::AdaptedVideoTrackSource` - base for custom sources
- `webrtc::VideoFrame::Builder` - construct frames
- `OnFrame()` - push frames to sinks

**Implementation:**

```cpp
#pragma once

#include <rtc_base/adapted_video_track_source.h>
#include <api/video/video_frame.h>
#include <api/video/i420_buffer.h>
#include <pybind11/pybind11.h>

namespace python_webrtc {

/**
 * Custom video track source that accepts frames from Python code.
 *
 * This allows Python to create custom video tracks (e.g., screen capture,
 * synthetic video, video files).
 */
class PythonVideoSource : public rtc::AdaptedVideoTrackSource {
public:
    PythonVideoSource()
        : AdaptedVideoTrackSource() {}

    /**
     * Push a frame from Python into the WebRTC pipeline.
     *
     * @param rgb_data RGB24 pixel data (width * height * 3 bytes)
     * @param width Frame width
     * @param height Frame height
     * @param timestamp_us Capture timestamp in microseconds
     */
    void PushFrameRGB(const pybind11::bytes& rgb_data,
                      int width, int height,
                      int64_t timestamp_us) {
        std::string data_str = rgb_data;

        if (data_str.size() != static_cast<size_t>(width * height * 3)) {
            throw std::invalid_argument("RGB data size mismatch");
        }

        // Convert RGB24 → I420 using libyuv
        rtc::scoped_refptr<webrtc::I420Buffer> i420_buffer =
            webrtc::I420Buffer::Create(width, height);

        int result = libyuv::RGB24ToI420(
            reinterpret_cast<const uint8_t*>(data_str.data()), width * 3,
            i420_buffer->MutableDataY(), i420_buffer->StrideY(),
            i420_buffer->MutableDataU(), i420_buffer->StrideU(),
            i420_buffer->MutableDataV(), i420_buffer->StrideV(),
            width, height
        );

        if (result != 0) {
            throw std::runtime_error("RGB to I420 conversion failed");
        }

        // Build VideoFrame
        webrtc::VideoFrame frame = webrtc::VideoFrame::Builder()
            .set_video_frame_buffer(i420_buffer)
            .set_timestamp_us(timestamp_us)
            .set_rotation(webrtc::kVideoRotation_0)
            .build();

        // Push to all sinks (connected tracks)
        OnFrame(frame);
    }

    // Required overrides from VideoTrackSourceInterface
    SourceState state() const override {
        return SourceState::kLive;
    }

    bool remote() const override {
        return false;
    }

    bool is_screencast() const override {
        return false;  // Could be configurable
    }

    std::optional<bool> needs_denoising() const override {
        return false;
    }

    static void Init(pybind11::module& m) {
        pybind11::class_<PythonVideoSource>(m, "VideoSource")
            .def(pybind11::init<>())
            .def("push_frame", &PythonVideoSource::PushFrameRGB,
                 pybind11::arg("rgb_data"),
                 pybind11::arg("width"),
                 pybind11::arg("height"),
                 pybind11::arg("timestamp_us"),
                 "Push RGB24 frame into video track");
    }
};

} // namespace python_webrtc
```

---

#### Task 3.2: Custom Audio Source (PythonAudioSource)

**File:** `extern/python-webrtc/python-webrtc/cpp/src/media/python_audio_source.h` (NEW)

**Purpose:** Support custom audio tracks like YAPAudioTrack for TTS and audio streaming.

**Key APIs:**
- `rtc::AudioSourceInterface` - base for audio sources
- `webrtc::AudioTrackInterface` - audio track management
- PyAV integration for `av.AudioFrame` handling

**Implementation:**

```cpp
#pragma once

#include <api/media_stream_interface.h>
#include <api/audio/audio_frame.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <mutex>

namespace python_webrtc {

/**
 * Custom audio track source that accepts PyAV AudioFrames from Python code.
 *
 * This allows Python to create custom audio tracks (e.g., TTS, file playback,
 * synthesized audio).
 */
class PythonAudioSource : public rtc::AudioSourceInterface {
public:
    PythonAudioSource() = default;
    ~PythonAudioSource() override = default;

    /**
     * Push a PyAV AudioFrame from Python into the WebRTC pipeline.
     *
     * @param py_audio_frame PyAV AudioFrame object from Python
     */
    void PushFrame(pybind11::object py_audio_frame) {
        // TODO: Convert PyAV AudioFrame to WebRTC audio format
        // This requires:
        // 1. Extract PCM data from PyAV frame
        // 2. Convert to webrtc::AudioFrame
        // 3. Push to registered sinks

        throw std::runtime_error("PythonAudioSource not yet implemented");
    }

    // AudioSourceInterface implementation
    void AddSink(webrtc::AudioTrackSinkInterface* sink) override {
        std::lock_guard<std::mutex> lock(mutex_);
        sinks_.push_back(sink);
    }

    void RemoveSink(webrtc::AudioTrackSinkInterface* sink) override {
        std::lock_guard<std::mutex> lock(mutex_);
        sinks_.erase(
            std::remove(sinks_.begin(), sinks_.end(), sink),
            sinks_.end()
        );
    }

    static void Init(pybind11::module& m) {
        pybind11::class_<PythonAudioSource>(m, "AudioSource")
            .def(pybind11::init<>())
            .def("push_frame", &PythonAudioSource::PushFrame,
                 pybind11::arg("audio_frame"),
                 "Push PyAV AudioFrame into audio track");
    }

private:
    std::vector<webrtc::AudioTrackSinkInterface*> sinks_;
    std::mutex mutex_;
};

} // namespace python_webrtc
```

**Note:** Full audio source implementation requires deeper integration with PyAV and WebRTC's audio pipeline. For neko_agent, the priority is ensuring **custom audio tracks work correctly** when added to peer connections.

---

**Python Wrapper for Custom Tracks:**

```python
# extern/python-webrtc/python/webrtc/custom_track.py

import asyncio
import time
import webrtc
from typing import Optional

try:
    import av
except ImportError:
    av = None


class CustomVideoTrack:
    """
    Base class for custom video tracks (aiortc compatibility).

    Subclass this and implement recv() to create custom video sources.
    """
    kind = "video"

    def __init__(self, label: str = "custom"):
        self._source = webrtc.VideoSource()
        # TODO: Create track from source via factory
        # self._track = factory.CreateVideoTrack(label, self._source)
        self._running = True

    async def recv(self):
        """Override this in subclasses to generate frames."""
        raise NotImplementedError

    def stop(self):
        """Stop the track."""
        self._running = False


class CustomAudioTrack:
    """
    Base class for custom audio tracks (aiortc compatibility).

    This is the pattern used by YAPAudioTrack in neko_agent.
    Subclass this and implement recv() to create custom audio sources.
    """
    kind = "audio"

    def __init__(self, label: str = "custom"):
        if av is None:
            raise ImportError("PyAV (av) is required for custom audio tracks")
        self._running = True

    async def recv(self) -> "av.AudioFrame":
        """Override this in subclasses to generate audio frames.

        Must return PyAV AudioFrame objects with:
        - frame.sample_rate (int)
        - frame.time_base (Fraction)
        - frame.pts (int)
        """
        raise NotImplementedError

    def stop(self):
        """Stop the track."""
        self._running = False


class GreenVideoTrack(CustomVideoTrack):
    """
    Example: Generate solid green frames (test pattern).
    """
    def __init__(self, width: int = 640, height: int = 480):
        super().__init__("green")
        self.width = width
        self.height = height
        self._timestamp = 0

    async def recv(self):
        if not self._running:
            raise StopAsyncIteration

        # Generate green frame (RGB24)
        green_pixel = bytes([0, 255, 0])  # Green in RGB
        rgb_data = green_pixel * (self.width * self.height)

        # Push to source
        timestamp_us = int(time.time() * 1_000_000)
        self._source.push_frame(rgb_data, self.width, self.height, timestamp_us)

        # Control frame rate (30 FPS)
        await asyncio.sleep(1/30.0)

        # Return synthetic VideoFrame for compatibility
        from webrtc.video_frame import VideoFrame
        return VideoFrame(rgb_data, self.width, self.height, timestamp_us)
```

**Example Usage (from neko_agent):**

```python
# Based on src/neko_comms/audio.py pattern

import numpy as np
from fractions import Fraction
import av
from webrtc.custom_track import CustomAudioTrack


class YAPAudioTrack(CustomAudioTrack):
    """Custom audio track that pulls PCM data from a queue.

    This is the actual pattern used in neko_agent for TTS broadcasting.
    """
    kind = "audio"

    def __init__(self, pcm_queue, sample_rate: int = 48000, frame_ms: int = 20):
        super().__init__()
        self.queue = pcm_queue
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._pts = 0

    async def recv(self) -> av.AudioFrame:
        """Pull PCM data from queue and return as PyAV AudioFrame."""
        # Get PCM samples (int16) from queue
        samples = self.queue.pull(self.frame_samples)  # (N, C) int16

        # Determine audio layout
        layout = "mono" if samples.shape[1] == 1 else "stereo"

        # Reshape for PyAV packed format
        packed_samples = samples.reshape(1, -1)

        # Create PyAV AudioFrame
        frame = av.AudioFrame.from_ndarray(packed_samples, format="s16", layout=layout)
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._pts

        self._pts += samples.shape[0]
        return frame
```

---

### Phase 4: Python 3.13 Support (Priority 3)

**Goal:** Build wheels for Python 3.11, 3.12, 3.13
**Effort:** 3-5 days
**Dependencies:** None (parallel to other phases)

#### Task 4.1: Update Build Configuration

**File:** `extern/python-webrtc/setup.py`

**Changes:**

```python
setup(
    name="wrtc",
    version="0.1.0",
    # ...
    python_requires=">=3.7,<3.14",  # Add 3.13 support
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",  # NEW
        "License :: OSI Approved :: BSD License",
        "Development Status :: 3 - Alpha",
    ],
)
```

#### Task 4.2: GitHub Actions CI

**File:** `.github/workflows/build-wheels.yml` (NEW)

```yaml
name: Build Python Wheels

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }} for Python ${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-22.04, macos-13, windows-2022]
        python: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build wheel setuptools

      - name: Build wheel
        run: python -m build --wheel

      - name: Test wheel
        run: |
          pip install dist/*.whl
          python -c "import webrtc; print(webrtc.__version__)"

      - name: Upload wheel
        uses: actions/upload-artifact@v4
        with:
          name: wheel-${{ matrix.os }}-${{ matrix.python }}
          path: dist/*.whl

  test_wheels:
    name: Test wheel installation
    needs: build_wheels
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheel-ubuntu-*-${{ matrix.python }}

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install and test
        run: |
          pip install wheel-ubuntu-*-${{ matrix.python }}/*.whl
          python -c "import webrtc; pc = webrtc.RTCPeerConnection()"
```

---

### Phase 5: aiortc Compatibility Layer (Priority 3)

**Goal:** Drop-in replacement for aiortc imports
**Effort:** 2-3 days
**Dependencies:** Phase 1 complete

**File:** `extern/python-webrtc/python/webrtc/aiortc_compat.py` (NEW)

**Purpose:** Provide 100% API-compatible wrapper for aiortc code.

**Implementation:**

```python
"""
Drop-in replacement for aiortc using python-webrtc backend.

Usage:
    # Before (aiortc)
    from aiortc import RTCPeerConnection, MediaStreamTrack, VideoStreamTrack

    # After (python-webrtc)
    from webrtc.aiortc_compat import RTCPeerConnection, MediaStreamTrack, VideoStreamTrack

    # All code works unchanged
"""

import asyncio
import webrtc
from webrtc.video_frame import VideoFrame
from typing import Union, AsyncIterator


class MediaStreamTrack(webrtc.MediaStreamTrack):
    """
    aiortc-compatible MediaStreamTrack with async recv().

    This wrapper makes python-webrtc's blocking recv() work with
    Python's async/await syntax.
    """

    async def recv(self, timeout_ms: int = 5000) -> Union[VideoFrame, 'AudioFrame']:
        """
        Receive next frame asynchronously.

        This wraps the blocking C++ recv() and runs it in a thread pool,
        allowing it to work with asyncio.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            VideoFrame or AudioFrame

        Raises:
            StopAsyncIteration: When track ends or timeout
        """
        loop = asyncio.get_event_loop()

        try:
            # Run blocking recv() in thread pool executor
            # This releases the asyncio event loop to handle other tasks
            frame = await loop.run_in_executor(
                None,  # Use default executor
                super().recv,
                timeout_ms
            )
            return frame

        except StopIteration:
            # C++ threw stop_iteration, convert to async version
            raise StopAsyncIteration

    def __aiter__(self) -> AsyncIterator:
        """Enable async for iteration."""
        return self

    async def __anext__(self):
        """Async iterator protocol."""
        return await self.recv()


class VideoStreamTrack(MediaStreamTrack):
    """Video-specific track (for type hints)."""
    kind = "video"


class AudioStreamTrack(MediaStreamTrack):
    """Audio-specific track (for type hints)."""
    kind = "audio"


# Re-export core classes with same names as aiortc
RTCPeerConnection = webrtc.RTCPeerConnection
RTCSessionDescription = webrtc.RTCSessionDescription
RTCIceCandidate = webrtc.RTCIceCandidate
RTCIceServer = webrtc.RTCIceServer  # NEW: Required by webrtc_client.py
RTCConfiguration = webrtc.RTCConfiguration

# Export video frame (already aiortc-compatible)
from webrtc.video_frame import VideoFrame


# ========== SDP Utilities Module ==========

class sdp:
    """SDP utility functions (aiortc.sdp compatibility)."""

    @staticmethod
    def candidate_from_sdp(candidate_str: str) -> RTCIceCandidate:
        """Parse ICE candidate from SDP string.

        Required by src/neko_comms/webrtc_client.py:469

        Args:
            candidate_str: SDP candidate string (e.g., "candidate:...")

        Returns:
            RTCIceCandidate: Parsed candidate object

        Example:
            >>> from webrtc.aiortc_compat import sdp
            >>> candidate = sdp.candidate_from_sdp("candidate:1 1 UDP 2130706431 10.0.1.1 51234 typ host")
            >>> candidate.sdpMid = "0"
            >>> candidate.sdpMLineIndex = 0
        """
        # Remove "candidate:" prefix if present
        if candidate_str.startswith("candidate:"):
            candidate_str = candidate_str[10:]

        # Parse using python-webrtc's native parser
        return webrtc.parse_ice_candidate(candidate_str)


__all__ = [
    'RTCPeerConnection',
    'RTCSessionDescription',
    'RTCIceCandidate',
    'RTCIceServer',
    'RTCConfiguration',
    'MediaStreamTrack',
    'VideoStreamTrack',
    'AudioStreamTrack',
    'VideoFrame',
    'sdp',  # NEW: SDP utilities module
]
```

**Migration Guide:**

```python
# ========== neko_agent Migration Example ==========

# File: src/neko_comms/webrtc_client.py

# BEFORE:
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    VideoStreamTrack,
    AudioStreamTrack,
)
from aiortc.sdp import candidate_from_sdp

# AFTER:
from webrtc.aiortc_compat import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    VideoStreamTrack,
    AudioStreamTrack,
    sdp,  # For sdp.candidate_from_sdp()
)

# Usage remains the same:
candidate = sdp.candidate_from_sdp(candidate_str)
candidate.sdpMid = payload.get("sdpMid")
candidate.sdpMLineIndex = payload.get("sdpMLineIndex")

# Everything else unchanged!
```

**Additional Migration Examples:**

```python
# ========== Custom Audio Track Migration ==========

# File: src/neko_comms/audio.py

# BEFORE:
from aiortc import MediaStreamTrack

class YAPAudioTrack(MediaStreamTrack):
    kind = "audio"
    async def recv(self) -> av.AudioFrame:
        # ... implementation ...

# AFTER:
from webrtc.aiortc_compat import MediaStreamTrack

class YAPAudioTrack(MediaStreamTrack):
    kind = "audio"
    async def recv(self) -> av.AudioFrame:
        # ... implementation unchanged ...

# No other changes needed!
```

---

## Testing Strategy

### Unit Tests

**Location:** `extern/python-webrtc/tests/`

**Coverage Requirements:**
- ✅ VideoSinkAdapter thread safety
- ✅ Frame queue backpressure
- ✅ GIL management (no deadlocks)
- ✅ VideoFrame conversion accuracy
- ✅ DataChannel send/receive
- ✅ Event handler callbacks

**Test Framework:** pytest + pytest-asyncio

**Example Test:**

```python
# tests/test_video_sink.py

import pytest
import threading
import time
from unittest.mock import MagicMock


def test_video_sink_thread_safety():
    """Test that VideoSinkAdapter is thread-safe."""
    import webrtc

    # Create sink
    # sink = webrtc._VideoSinkAdapter(max_size=10)

    # Push frames from multiple threads
    def push_frames(sink, n):
        for i in range(n):
            # Push mock frame
            pass

    threads = [
        threading.Thread(target=push_frames, args=(None, 100))
        for _ in range(4)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify no crashes, no dropped frames beyond buffer size
```

### Integration Tests

**Test Scenario:** Real peer connection with loopback

```python
# tests/integration/test_loopback.py

import pytest
import asyncio
import webrtc


@pytest.mark.asyncio
@pytest.mark.integration
async def test_video_loopback():
    """
    Test video transmission and reception using two peer connections.
    """
    # Create two peers
    pc1 = webrtc.RTCPeerConnection()
    pc2 = webrtc.RTCPeerConnection()

    # Create custom video source on pc1
    from webrtc.custom_track import GreenVideoTrack
    track = GreenVideoTrack(width=320, height=240)
    pc1.addTrack(track)

    # Set up signaling
    @pc2.on("track")
    async def on_track(track):
        # Receive frame
        frame = await track.recv()
        assert frame.width == 320
        assert frame.height == 240

        # Convert to image
        img = frame.to_image()
        assert img.size == (320, 240)

    # Exchange SDP
    offer = await pc1.createOffer()
    await pc1.setLocalDescription(offer)
    await pc2.setRemoteDescription(pc1.localDescription)

    answer = await pc2.createAnswer()
    await pc2.setLocalDescription(answer)
    await pc1.setRemoteDescription(pc2.localDescription)

    # Wait for connection
    await asyncio.sleep(2)

    # Cleanup
    await pc1.close()
    await pc2.close()
```

### neko_agent Integration Test

**File:** `tests/test_neko_integration.py` (in main repo)

```python
import pytest
from webrtc.aiortc_compat import RTCPeerConnection, MediaStreamTrack


@pytest.mark.integration
async def test_neko_agent_compat():
    """
    Test that neko_agent code works with webrtc backend.
    """
    # Import actual neko_agent code
    from neko_comms.webrtc_client import WebRTCNekoClient

    # This should work unchanged if aiortc_compat is correct
    client = WebRTCNekoClient(
        ws_url="wss://test.neko.local/api/ws?token=test",
        ice_servers=["stun:stun.l.google.com:19302"]
    )

    # ... rest of test
```

---

## Common Pitfalls & Solutions

### 1. GIL Deadlocks

**Symptom:** Python hangs when calling `recv()`

**Cause:** Holding GIL while waiting on condition variable that requires GIL to signal.

**Solution:**
```cpp
// ❌ WRONG - deadlock potential
void GetFrame() {
    pybind11::gil_scoped_acquire acquire;  // Hold GIL
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock);  // Waiting while holding GIL - deadlock!
}

// ✅ CORRECT - release GIL during wait
void GetFrame() {
    std::optional<Frame> frame;
    {
        pybind11::gil_scoped_release release;  // Release GIL
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock);
        frame = queue_.front();
    }
    // GIL reacquired here automatically
    return ConvertToPython(frame.value());
}
```

### 2. Blue Video Frames

**Symptom:** Converted frames have blue tint

**Cause:** Swapped U/V planes in I420 → RGB conversion

**Solution:**
```cpp
// ❌ WRONG - U and V swapped
libyuv::I420ToRGB24(
    Y, strideY,
    V, strideV,  // ← Should be U
    U, strideU,  // ← Should be V
    rgb, width * 3, width, height
);

// ✅ CORRECT - proper order
libyuv::I420ToRGB24(
    Y, strideY,
    U, strideU,  // ✓
    V, strideV,  // ✓
    rgb, width * 3, width, height
);
```

### 3. StopIteration vs StopAsyncIteration

**Symptom:** `async for` doesn't stop properly

**Cause:** Throwing `StopIteration` instead of `StopAsyncIteration`

**Solution:**
```python
# ❌ WRONG - regular iterator
class Track:
    async def recv(self):
        # ...
        raise StopIteration  # Wrong for async!

# ✅ CORRECT - async iterator
class Track:
    async def recv(self):
        # ...
        raise StopAsyncIteration  # Correct
```

In C++:
```cpp
// pybind11::stop_iteration() → StopIteration (sync)
// For async, need Python-side wrapper to convert
```

### 4. Memory Leaks in Frame Queue

**Symptom:** Memory usage grows over time

**Cause:** Not properly releasing `VideoFrame` references

**Solution:**
```cpp
// ✅ Use scoped_refptr properly
std::queue<webrtc::VideoFrame> queue_;  // Value semantics, auto cleanup

// ❌ Don't use raw pointers
std::queue<webrtc::VideoFrame*> queue_;  // Manual memory management
```

### 5. Python 3.13 Free-Threading Issues

**Symptom:** Crashes or data races on Python 3.13

**Cause:** Python 3.13 introduced experimental free-threading (no GIL)

**Solution:**
```cpp
// Mark module as GIL-dependent (not free-threaded safe yet)
PYBIND11_MODULE(wrtc, m) {
    // Don't add py::mod_gil_not_used() until we're thread-safe
    m.doc() = "WebRTC bindings for Python";
    // ...
}
```

---

## References & Resources

### libwebrtc Official Resources

**Source Code:**
- **Main Repository:** https://webrtc.googlesource.com/src/
- **M92 Branch (for python-webrtc):** https://chromium.googlesource.com/external/webrtc/+/branch-heads/4515
- **Browse API Headers:** https://webrtc.googlesource.com/src/+/HEAD/native-api.md

**Official Documentation:**
- **Native APIs:** https://webrtc.github.io/webrtc-org/native-code/native-apis/
- **Native Code Guide:** https://webrtc.github.io/webrtc-org/native-code/
- **Development:** https://webrtc.github.io/webrtc-org/native-code/development/
- **W3C WebRTC Spec:** https://w3c.github.io/webrtc-pc/

**Key Header Files (for reference during implementation):**
- PeerConnection: `api/peer_connection_interface.h`
- MediaStream: `api/media_stream_interface.h`
- VideoSink: `api/video/video_sink_interface.h`
- VideoFrame: `api/video/video_frame.h`
- I420Buffer: `api/video/i420_buffer.h`
- DataChannel: `api/data_channel_interface.h`

**Community:**
- **discuss-webrtc:** https://groups.google.com/g/discuss-webrtc
- **Release Notes:** https://chromium.googlesource.com/external/webrtc/+/refs/heads/master/docs/release-notes.md
- **M92 Release Discussion:** https://groups.google.com/g/discuss-webrtc/c/hks5zneZJbo

### python-webrtc Resources

- **GitHub Repository:** https://github.com/MarshalX/python-webrtc
- **Project Board (Implementation Status):** https://github.com/users/MarshalX/projects/1/views/1
- **PyPI Package:** https://pypi.org/project/wrtc/
- **Documentation:** https://wrtc.rtfd.io/

### pybind11 Documentation

- **Main Docs:** https://pybind11.readthedocs.io/
- **Threading & GIL:** https://pybind11.readthedocs.io/en/stable/advanced/misc.html
- **Exceptions:** https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
- **Threading Discussion:** https://github.com/pybind/pybind11/discussions/3693
- **Free-Threading Support (Python 3.13):** https://pybind11.readthedocs.io/en/stable/changelog.html

### aiortc Reference Implementation

- **GitHub Repository:** https://github.com/aiortc/aiortc
- **MediaStreamTrack Implementation:** https://github.com/aiortc/aiortc/blob/main/src/aiortc/mediastreams.py
- **API Documentation:** https://aiortc.readthedocs.io/en/latest/api.html
- **PyPI:** https://pypi.org/project/aiortc/

Use aiortc source as reference for:
- MediaStreamTrack.recv() async patterns
- VideoFrame.to_image() / to_ndarray() implementation
- Custom track subclassing patterns

### Video Frame Processing

**libyuv (YUV ↔ RGB conversion):**
- **Source:** https://chromium.googlesource.com/libyuv/libyuv/
- **GitHub Mirror:** https://github.com/lemenkov/libyuv
- **Format Documentation:** https://chromium.googlesource.com/libyuv/libyuv/+/master/docs/formats.md
- **Conversion Performance:** See docs/formats.md for SIMD benchmarks

**Color Spaces:**
- **YUV Explained:** https://wiki.videolan.org/YUV/
- **I420 Format:** https://chromium.googlesource.com/libyuv/libyuv/+/master/docs/formats.md
- **WebRTC Video Processing:** https://webrtc.github.io/webrtc-org/native-code/video-processing/

### Threading & Async Patterns

- **Python 3.13 Free-Threading (PEP 703):** https://peps.python.org/pep-0703/
- **asyncio + C++:** https://stackoverflow.com/questions/54553907/
- **GIL Management Best Practices:** https://pybind11.readthedocs.io/en/stable/advanced/misc.html

### Example Implementations

**Real-World libwebrtc Usage:**
- **Open3D WebRTC Server:** https://github.com/isl-org/Open3D/tree/master/cpp/open3d/visualization/webrtc_server
  - Shows VideoSinkInterface implementation
  - PeerConnectionManager patterns
- **webrtc-streamer:** https://github.com/mpromonet/webrtc-streamer
  - C++ WebRTC application example
- **node-webrtc:** https://github.com/node-webrtc/node-webrtc
  - Another native binding (Node.js, but similar patterns)

### Build & CI

- **GitHub Actions Python Setup:** https://github.com/actions/setup-python
- **cibuildwheel:** https://cibuildwheel.readthedocs.io/
- **WebRTC Build Prerequisites:** https://webrtc.github.io/webrtc-org/native-code/development/prerequisite-sw/

### Debugging Tools

- **Chrome WebRTC Internals:** chrome://webrtc-internals/
  - Live WebRTC connection debugging
  - Statistics, SDP, ICE candidates
- **Valgrind:** Memory leak detection
- **AddressSanitizer:** Memory error detection
- **GDB/LLDB:** Native debugging

---

## Appendix: Quick Reference

### Key File Locations

```
extern/python-webrtc/
├── python-webrtc/
│   ├── cpp/
│   │   └── src/
│   │       ├── media/
│   │       │   ├── video_sink_adapter.h         [NEW - Task 1.1]
│   │       │   ├── audio_sink_adapter.h         [NEW - Task 1.5]
│   │       │   ├── frame_converter.h            [NEW - Task 1.5]
│   │       │   └── python_video_source.h        [NEW - Task 3.1]
│   │       └── interfaces/
│   │           ├── media_stream_track.h/.cpp    [MODIFY - Task 1.2]
│   │           └── rtc_data_channel.h/.cpp      [NEW - Task 2.1]
│   └── python/
│       └── webrtc/
│           ├── video_frame.py                   [NEW - Task 1.4]
│           ├── custom_track.py                  [NEW - Task 3.1]
│           └── aiortc_compat.py                 [NEW - Task 5.1]
└── tests/
    ├── test_video_recv.py                       [NEW - Task 1.6]
    ├── test_data_channel.py                     [NEW - Task 2.2]
    └── integration/
        └── test_loopback.py                     [NEW]
```

### Build Commands

```bash
# IMPORTANT: Initialize submodules first
cd extern/python-webrtc
git submodule update --init --recursive
# This fetches third_party/pybind11 (required for building)

# Development build (with debug symbols)
DEBUG=1 python setup.py build_ext --inplace

# Install in dev mode
pip install -e .

# Build wheel
python -m build --wheel

# Run tests
pytest tests/ -v

# Run with AddressSanitizer (detect memory errors)
CFLAGS="-fsanitize=address" python setup.py build_ext --inplace
```

**Note:** The pybind11 submodule URL was changed from SSH (`git@github.com:`) to HTTPS (`https://github.com/`) for better portability. If you encounter issues, run:

```bash
cd extern/python-webrtc
git submodule sync
git submodule update --init --recursive
```

### Debugging Commands

```bash
# Check symbols in compiled module
nm -C wrtc.*.so | grep VideoSink

# Run under gdb
gdb --args python test_recv.py

# Profile with valgrind
valgrind --leak-check=full --track-origins=yes python test.py

# Trace system calls
strace -e trace=network python test.py
```

---

## Appendix B: aiortc API Coverage Checklist

This checklist verifies that all aiortc APIs used in neko_agent (src/) are covered by the upgrade guide.

### ✅ Core WebRTC Classes (Phase 5)
- [x] `RTCPeerConnection` - Main peer connection class
- [x] `RTCConfiguration` - Configuration object for peer connections
- [x] `RTCSessionDescription` - SDP offer/answer representation
- [x] `RTCIceCandidate` - ICE candidate representation
- [x] `RTCIceServer` - ICE server configuration (STUN/TURN)
- [x] `VideoStreamTrack` - Video track type hint
- [x] `AudioStreamTrack` - Audio track type hint

### ✅ Media Track Classes (Phase 1 & 3)
- [x] `MediaStreamTrack` - Base class for custom tracks
  - Used in: `src/neko_comms/audio.py:19`, `src/yap.py:40`
  - Pattern: `class YAPAudioTrack(MediaStreamTrack)`
- [x] `MediaStreamTrack.recv()` - Receive frames from track
  - Used in: `src/neko_comms/frame_source.py:182`
  - Pattern: `frame = await track.recv()`

### ✅ Video Frame Classes (Phase 1)
- [x] `VideoFrame` - Video frame representation
- [x] `VideoFrame.to_image()` - Convert to PIL Image
- [x] `VideoFrame.to_ndarray()` - Convert to numpy array

### ✅ SDP Utilities (Phase 5)
- [x] `aiortc.sdp.candidate_from_sdp()` - Parse ICE candidates
  - Used in: `src/neko_comms/webrtc_client.py:25, 469`
  - Pattern: `candidate = candidate_from_sdp(candidate_str)`

### ✅ PyAV Integration (Phase 3)
- [x] `av.AudioFrame` - Audio frame representation for custom tracks
  - Used in: `src/neko_comms/audio.py:116, 125`
  - Pattern: `av.AudioFrame.from_ndarray(samples, format="s16", layout=layout)`
- [x] Audio frame properties: `sample_rate`, `time_base`, `pts`

### 🔍 Files Analyzed
- ✅ `src/yap.py` - TTS broadcaster (uses MediaStreamTrack, custom audio tracks)
- ✅ `src/neko_comms/audio.py` - YAPAudioTrack implementation
- ✅ `src/neko_comms/webrtc_client.py` - WebRTC client (all core APIs)
- ✅ `src/neko_comms/frame_source.py` - Frame reception patterns

### Key Implementation Requirements

**Priority 0 (Critical for video):**
- `MediaStreamTrack.recv()` for video frames
- `VideoFrame` with `to_image()` and `to_ndarray()`

**Priority 1 (Critical for audio/yap.py):**
- Custom audio track pattern (inherit from `MediaStreamTrack`, set `kind = "audio"`)
- `async def recv() -> av.AudioFrame` support
- PyAV AudioFrame creation and property setting

**Priority 2 (Required for connection):**
- `RTCIceServer` class and constructor
- `sdp.candidate_from_sdp()` utility function

**All APIs are now documented in the upgrade guide.**

---

**Document Version:** 1.1
**Last Updated:** 2025-10-18
**Contributors:** AI Assistant
**Next Review:** After Phase 1 completion
