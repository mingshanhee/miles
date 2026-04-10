"""Integration tests for session HTTP routes (create / get / delete / proxy)."""

import re
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from miles.rollout.session.session_server import SessionServer
from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@pytest.fixture(scope="class")
def router_env():
    """Create a standalone SessionServer with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
        }
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with with_mock_server(process_fn=process_fn) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint="Qwen/Qwen3-0.6B",
                chat_template_path=None,
                trajectory_manager="linear_trajectory",
                session_server_instance_id=uuid.uuid4().hex,
            )
            server_obj = SessionServer(args, backend_url=backend.url)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()

            url = f"http://127.0.0.1:{port}"

            try:
                yield SimpleNamespace(url=url, backend=backend)
            finally:
                server.stop()


class TestSessionRoutes:
    def test_health_reports_stable_instance_id(self, router_env):
        first = requests.get(f"{router_env.url}/health", timeout=5.0)
        second = requests.get(f"{router_env.url}/health", timeout=5.0)

        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        second_body = second.json()
        assert first_body["status"] == "ok"
        assert second_body["status"] == "ok"
        assert re.fullmatch(r"[0-9a-f]{32}", first_body["session_server_instance_id"])
        assert second_body["session_server_instance_id"] == first_body["session_server_instance_id"]

    def test_create_session(self, router_env):
        response = requests.post(f"{router_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_env):
        response = requests.get(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"

    def test_delete_session(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"


class TestSessionProxy:
    def test_proxy_chat_appends_record(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["status_code"] == 200

    def test_proxy_chat_sets_session_routing_key_header(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        assert router_env.backend.request_headers_log
        assert router_env.backend.request_headers_log[-1].get("x-smg-routing-key") == session_id


def test_session_affinity_stays_on_same_worker_when_router_load_changes():
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    router_args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=find_available_port(32000),
        rollout_health_check_interval=1.0,
        miles_router_health_check_failure_threshold=3,
        miles_router_max_connections=100,
        miles_router_timeout=None,
        miles_router_middleware_paths=[],
    )
    router = MilesRouter(router_args, verbose=False)
    router_server = UvicornThreadServer(router.app, host=router_args.sglang_router_ip, port=router_args.sglang_router_port)

    with with_mock_server(process_fn=process_fn, port=find_available_port(33000)) as worker1:
        with with_mock_server(process_fn=process_fn, port=find_available_port(33100)) as worker2:
            router_server.start()
            try:
                requests.post(f"{router_server.url}/add_worker", params={"url": worker1.url}, timeout=5.0).raise_for_status()
                requests.post(f"{router_server.url}/add_worker", params={"url": worker2.url}, timeout=5.0).raise_for_status()

                session_args = SimpleNamespace(
                    miles_router_timeout=30,
                    hf_checkpoint="Qwen/Qwen3-0.6B",
                    chat_template_path=None,
                    trajectory_manager="linear_trajectory",
                    session_server_instance_id=uuid.uuid4().hex,
                )
                session_server = SessionServer(session_args, backend_url=router_server.url)
                session_port = find_available_port(34000)
                session_uvicorn = UvicornThreadServer(session_server.app, host="127.0.0.1", port=session_port)
                session_uvicorn.start()
                try:
                    session_url = f"http://127.0.0.1:{session_port}"
                    session_id = requests.post(f"{session_url}/sessions", timeout=5.0).json()["session_id"]
                    payload = {
                        "messages": [{"role": "user", "content": "What is 1+2?"}],
                        "return_logprob": True,
                    }

                    first = requests.post(
                        f"{session_url}/sessions/{session_id}/v1/chat/completions",
                        json=payload,
                        timeout=10.0,
                    )
                    assert first.status_code == 200

                    def _keyed_hits(worker: MockSGLangServer) -> int:
                        return sum(
                            headers.get("x-smg-routing-key") == session_id for headers in worker.request_headers_log
                        )

                    if _keyed_hits(worker1):
                        selected_worker = worker1
                        other_worker = worker2
                    else:
                        selected_worker = worker2
                        other_worker = worker1

                    router.worker_request_counts[selected_worker.url] += 1
                    try:
                        second = requests.post(
                            f"{session_url}/sessions/{session_id}/v1/chat/completions",
                            json=payload,
                            timeout=10.0,
                        )
                        assert second.status_code == 200
                    finally:
                        router.worker_request_counts[selected_worker.url] -= 1

                    assert _keyed_hits(selected_worker) == 2
                    assert _keyed_hits(other_worker) == 0
                finally:
                    session_uvicorn.stop()
            finally:
                router_server.stop()
