"""Prometheus metrics module for Neko Agent.

This module provides centralized Prometheus metrics definitions and utilities
for monitoring Neko Agent operations across all components.
"""

import logging
from typing import Any, Tuple
from prometheus_client import start_http_server, Counter, Histogram

# Metrics definitions
frames_received = Counter("neko_frames_received_total", "Total video frames received")
actions_executed = Counter("neko_actions_executed_total", "Actions executed by type", ["action_type"])
parse_errors = Counter("neko_parse_errors_total", "Action parse errors")
navigation_steps = Counter("neko_navigation_steps_total", "Navigation step count")
inference_latency = Histogram("neko_inference_latency_seconds", "Inference latency")
reconnects = Counter("neko_reconnects_total", "WS reconnect attempts")
resize_duration = Histogram("neko_resize_duration_seconds", "Resize time")


def start_metrics_server(port: int, logger: logging.Logger) -> Tuple[Any, Any]:
    """Start Prometheus metrics server.

    :param port: Port number to bind the metrics server to
    :param logger: Logger instance for recording server startup status
    :return: Tuple of (server, thread) for clean shutdown
    """
    try:
        ret = start_http_server(port)
        if isinstance(ret, tuple) and len(ret) == 2:
            server, thread = ret
        else:
            server, thread = ret, None
        logger.info("Metrics server started on port %d", port)
        return server, thread
    except Exception as e:
        logger.error("Failed to start metrics server on port %d: %s", port, e)
        return None, None