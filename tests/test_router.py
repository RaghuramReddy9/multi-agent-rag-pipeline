"""Unit tests for the router agent.

Tests classification logic without calling the LLM API.
"""

from __future__ import annotations

import pytest

from router import classify_query, ROUTER_SYSTEM_PROMPT, DEPARTMENTS


class TestRouterConstants:
    def test_departments_tuple(self):
        assert DEPARTMENTS == ("billing", "technical", "general")

    def test_system_prompt_mentions_all_departments(self):
        for dept in DEPARTMENTS:
            assert dept in ROUTER_SYSTEM_PROMPT.lower()


class TestClassifyQuery:
    """Test router classification with mocked LLM responses.

    In a real setup you'd use unittest.mock to patch the Groq client.
    These tests verify the function signature and fallback behavior.
    """

    def test_fallback_on_error(self, monkeypatch):
        """If the LLM call raises, router should return 'general'."""
        import router

        def _raise(*args, **kwargs):
            raise Exception("API down")

        monkeypatch.setattr(router, "_get_client", lambda: type("C", (), {
            "chat": type("Chat", (), {
                "completions": type("Comp", (), {"create": _raise})()
            })()
        })())

        result = classify_query("test query")
        assert result == "general"

    def test_billing_keyword_routing(self, monkeypatch):
        """Router should return 'billing' when LLM output contains 'billing'."""
        import router

        fake_completion = type("Msg", (), {"content": "billing"})()
        fake_choice = type("Choice", (), {"message": fake_completion})()
        fake_resp = type("Resp", (), {"choices": [fake_choice]})()

        monkeypatch.setattr(router, "_get_client", lambda: type("C", (), {
            "chat": type("Chat", (), {
                "completions": type("Comp", (), {"create": lambda *a, **kw: fake_resp})()
            })()
        })())

        result = classify_query("I want a refund")
        assert result == "billing"

    def test_technical_keyword_routing(self, monkeypatch):
        """Router should return 'technical' when LLM output contains 'technical'."""
        import router

        fake_completion = type("Msg", (), {"content": "technical"})()
        fake_choice = type("Choice", (), {"message": fake_completion})()
        fake_resp = type("Resp", (), {"choices": [fake_choice]})()

        monkeypatch.setattr(router, "_get_client", lambda: type("C", (), {
            "chat": type("Chat", (), {
                "completions": type("Comp", (), {"create": lambda *a, **kw: fake_resp})()
            })()
        })())

        result = classify_query("My app crashed")
        assert result == "technical"

    def test_unknown_output_falls_back_to_general(self, monkeypatch):
        """If LLM returns something unexpected, fallback to 'general'."""
        import router

        fake_completion = type("Msg", (), {"content": "xyz_unknown"})()
        fake_choice = type("Choice", (), {"message": fake_completion})()
        fake_resp = type("Resp", (), {"choices": [fake_choice]})()

        monkeypatch.setattr(router, "_get_client", lambda: type("C", (), {
            "chat": type("Chat", (), {
                "completions": type("Comp", (), {"create": lambda *a, **kw: fake_resp})()
            })()
        })())

        result = classify_query("hello")
        assert result == "general"
