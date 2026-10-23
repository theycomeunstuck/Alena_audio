"""Real HTTP forwarding with a deliberate loss after an upstream response commits.

The proxy never supplies application responses.  An armed successful action is
read completely from the real API and recorded.  Its client socket then closes
either before headers or after a strict prefix of the real response body.
Subsequent retries pass through.
"""
from __future__ import annotations

from copy import deepcopy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from urllib.parse import urlsplit


_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "proxy-connection",
})


def _json_body(body: bytes):
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None


def _forward_headers(headers):
    connection_tokens = {
        token.strip().lower()
        for name, value in headers
        if name.lower() == "connection"
        for token in value.split(",")
    }
    excluded = _HOP_BY_HOP | connection_tokens | {"content-length"}
    return [(name, value) for name, value in headers if name.lower() not in excluded]


class ResponseLossProxy:
    """Context-managed HTTP proxy for an intentionally lost action response.

    ``arm()`` clears ``dropped`` and the previous capture.  Once ``dropped`` is
    set, ``captured`` contains the complete request/response JSON, response
    status, bytes and fault mode.  ``request_log`` contains metadata for all
    completed forwarded requests, excluding authorization headers.
    """

    def __init__(self, upstream_url: str, *, timeout: float = 60.0):
        upstream = urlsplit(upstream_url)
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError("upstream_url must be an HTTP or HTTPS URL")
        if upstream.query or upstream.fragment or upstream.username or upstream.password:
            raise ValueError("upstream_url must not contain credentials, query or fragment")
        self._upstream = upstream
        self._timeout = timeout
        self._lock = threading.Lock()
        self._armed_operation = None
        self._armed_mode = None
        self._captured = {}
        self._request_log = []
        self.dropped = threading.Event()
        self._server = None
        self._thread = None
        self.base_url = None

    @property
    def captured(self) -> dict:
        with self._lock:
            return deepcopy(self._captured)

    @property
    def request_log(self) -> list[dict]:
        with self._lock:
            return deepcopy(self._request_log)

    def arm(self, operation: str = "autonomous_division_answer", *,
            mode: str = "close_before_headers") -> None:
        """Lose one real successful response using the selected transport fault.

        ``close_before_headers`` closes without sending a response. Chromium
        may transparently retry that transport failure. ``truncate_body`` sends
        the real status and headers, advertises the full actual body length,
        then sends only its first byte before closing. No response bytes are
        invented in either mode.
        """
        if mode not in {"close_before_headers", "truncate_body"}:
            raise ValueError("unsupported response loss mode")
        with self._lock:
            if self._armed_operation is not None:
                raise RuntimeError("response loss is already armed")
            self._captured = {}
            self.dropped.clear()
            self._armed_operation = operation
            self._armed_mode = mode

    def __enter__(self):
        if self._server is not None:
            raise RuntimeError("proxy is already running")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _body(self):
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    chunks = []
                    while True:
                        line = self.rfile.readline()
                        size = int(line.split(b";", 1)[0].strip(), 16)
                        if not size:
                            while self.rfile.readline() not in {b"\r\n", b"\n", b""}:
                                pass
                            return b"".join(chunks)
                        chunk = self.rfile.read(size)
                        if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                            raise ValueError("incomplete chunked request")
                        chunks.append(chunk)
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                return body

            def _close_without_response(self):
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()

            def _forward(self):
                started = time.monotonic()
                request_json = None
                operation = None
                status = None
                dropped = False
                fault_mode = None
                error = None
                upstream_connection = None
                try:
                    body = self._body()
                    request_json = _json_body(body)
                    if isinstance(request_json, dict):
                        action = request_json.get("learning_action")
                        if isinstance(action, dict):
                            operation = action.get("operation")
                    headers = {
                        name: value for name, value in _forward_headers(list(self.headers.items()))
                        if name.lower() != "host"
                    }
                    headers["Host"] = proxy._upstream.netloc
                    headers["Content-Length"] = str(len(body))
                    connection_type = (http.client.HTTPSConnection
                                       if proxy._upstream.scheme == "https"
                                       else http.client.HTTPConnection)
                    upstream_connection = connection_type(
                        proxy._upstream.hostname, proxy._upstream.port,
                        timeout=proxy._timeout,
                    )
                    upstream_path = proxy._upstream.path.rstrip("/") + self.path
                    upstream_connection.request(self.command, upstream_path, body=body, headers=headers)
                    response = upstream_connection.getresponse()
                    status = response.status
                    response_headers = response.getheaders()
                    response_body = response.read()
                    matches = (self.command == "POST"
                               and urlsplit(self.path).path == "/v1/chat/completions"
                               and status == 200)
                    with proxy._lock:
                        if (matches and proxy._armed_operation is not None
                                and operation == proxy._armed_operation):
                            fault_mode = proxy._armed_mode
                            proxy._armed_operation = None
                            proxy._armed_mode = None
                            proxy._captured = {
                                "request_json": request_json,
                                "response_json": _json_body(response_body),
                                "status": status,
                                "response_body": response_body,
                                "response_headers": response_headers,
                                "method": self.command,
                                "path": self.path,
                                "mode": fault_mode,
                            }
                            dropped = True
                    if dropped:
                        if fault_mode == "truncate_body":
                            self.send_response_only(status, response.reason)
                            for name, value in _forward_headers(response_headers):
                                self.send_header(name, value)
                            self.send_header("Content-Length", str(len(response_body)))
                            self.end_headers()
                            self.wfile.write(response_body[:1])
                            self.wfile.flush()
                        self._close_without_response()
                        return
                    self.send_response_only(status, response.reason)
                    for name, value in _forward_headers(response_headers):
                        self.send_header(name, value)
                    if self.command == "HEAD" or status == 304:
                        content_length = response.getheader("Content-Length")
                        if content_length is not None:
                            self.send_header("Content-Length", content_length)
                    elif status != 204 and status >= 200:
                        self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    if self.command != "HEAD" and status not in {204, 304}:
                        self.wfile.write(response_body)
                        self.wfile.flush()
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self._close_without_response()
                finally:
                    if upstream_connection is not None:
                        upstream_connection.close()
                    metadata = {
                        "method": self.command,
                        "path": self.path,
                        "operation": operation,
                        "status": status,
                        "dropped": dropped,
                        "mode": fault_mode,
                        "duration_seconds": round(time.monotonic() - started, 6),
                    }
                    if isinstance(request_json, dict):
                        for key in ("client_message_id", "learning_session_id", "learner_id"):
                            if key in request_json:
                                metadata[key] = request_json[key]
                    if error:
                        metadata["error"] = error
                    with proxy._lock:
                        proxy._request_log.append(metadata)
                    if dropped:
                        proxy.dropped.set()

            do_GET = do_POST = do_PUT = do_PATCH = _forward
            do_DELETE = do_OPTIONS = do_HEAD = do_TRACE = _forward

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
            self._server = None
            self._thread = None
