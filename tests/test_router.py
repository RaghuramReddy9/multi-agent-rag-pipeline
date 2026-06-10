"""Unit tests for the router agent.

Tests classification logic without calling the LLM API.
Uses lazy imports to avoid triggering config loading at module level.
"""

from __future__ import annotations


class TestRouterConstants:
    def test_departments_tuple(self):
        from router import DEPARTMENTS
        assert DEPARTMENTS == ("billing", "technical", "general")

    def test_system_prompt_mentions_all_departments(self):
        from router import ROUTER_SYSTEM_PROMPT
        for dept in ("billing", "technical", "general"):
            assert dept in ROUTER_SYSTEM_PROMPT.lower()


class TestClassifyQuery:
    """Test router classification with mocked LLM responses."""

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

        from router import classify_query
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

        from router import classify_query
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

        from router import classify_query
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

        from router import classify_query
        result = classify_query("hello")
        assert result == "general"
