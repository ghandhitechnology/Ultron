import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from ultron.serving_router import ReplicaRouter, RouterServer


def test_affinity_spills_to_free_replica_and_releases_on_failure():
    router = ReplicaRouter(["http://127.0.0.1:8101", "http://127.0.0.1:8102"], [{"id": "frozen"}], skew=0)
    with router.lease("episode:frozen") as first:
        with pytest.raises(RuntimeError):
            with router.lease("episode:frozen") as second:
                assert second != first
                raise RuntimeError("client disconnected")
        assert sum(router.active) == 1
    assert router.active == [0, 0]
    with router.lease("episode:frozen") as again:
        assert again == first


def test_router_preserves_sampling_and_stream_and_rejects_unknown_model():
    requests = []
    stream = b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\ndata: [DONE]\n\n'

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            body = b'{"data":[{"id":"attacker-frozen"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            requests.append((json.loads(self.rfile.read(int(self.headers["Content-Length"]))), self.headers["X-Ultron-Episode"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(stream)))
            self.end_headers()
            self.wfile.write(stream)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    router = ReplicaRouter([f"http://127.0.0.1:{upstream.server_port}"], [{"id": "attacker-frozen", "root": "/immutable/adapter"}])
    server = RouterServer(("127.0.0.1", 0), router)
    threads = [threading.Thread(target=value.serve_forever, daemon=True) for value in (upstream, server)]
    for thread in threads:
        thread.start()
    try:
        assert router.health()
        payload = {"model": "attacker-frozen", "messages": [{"role": "user", "content": "same prompt"}], "seed": 17, "temperature": 0.8, "stream": True}
        connection = http.client.HTTPConnection(*server.server_address)
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"X-Ultron-Episode": "42"})
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == stream
        connection.close()
        assert requests == [(payload, "42")]
        connection = http.client.HTTPConnection(*server.server_address)
        connection.request("POST", "/v1/chat/completions", json.dumps({"model": "mutable-latest"}))
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
        assert len(requests) == 1
        assert router.active == [0]
    finally:
        server.shutdown()
        upstream.shutdown()
        server.server_close()
        upstream.server_close()
