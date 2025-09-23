"""Authentication utilities for Neko servers.

This module provides standalone authentication functions for connecting
to Neko servers via REST API and obtaining WebSocket connection details.
"""

import requests
from typing import Tuple


def rest_login_and_ws_url(base_url: str, username: str, password: str) -> Tuple[str, str, str]:
    """Authenticate via REST and return WebSocket URL.

    :param base_url: Base HTTP URL (e.g., https://neko.example.com)
    :param username: Username for authentication
    :param password: Password for authentication
    :return: Tuple of (ws_url, base_url, token)
    :raises ValueError: If authentication fails
    """
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must start with http:// or https://")

    login_url = f"{base_url.rstrip('/')}/api/login"
    login_data = {"username": username, "password": password}

    try:
        response = requests.post(login_url, json=login_data, timeout=10)
        response.raise_for_status()
        data = response.json()
        token = data.get("token")
        if not token:
            raise ValueError("No token returned from login")
    except Exception as e:
        raise ValueError(f"Login failed: {e}")

    # Build WebSocket URL
    scheme = "wss" if base_url.startswith("https") else "ws"
    host = base_url.split("://", 1)[1]
    ws_url = f"{scheme}://{host}/api/ws?token={token}"

    return ws_url, base_url, token