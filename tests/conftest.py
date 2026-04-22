"""
Shared pytest fixtures for Sentinel tests.

sys.path is configured here once for all test files.
Individual test files must NOT add their own sys.path.insert calls.
"""
import sys
import os

# Add project root to path once — all test files rely on this via conftest.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.environment import GuardrailEnvironment


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fresh_env():
    return GuardrailEnvironment()


def first_prompt_id(client_fixture, task_id: str = "basic_threat_detection") -> str:
    return client_fixture.post(f"/reset?task_id={task_id}").json()["prompt_id"]
