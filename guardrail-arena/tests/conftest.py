"""
Shared pytest fixtures for Guardrail Arena tests.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
