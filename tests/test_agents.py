"""Tests for agent modules.

Verifies that agents handle errors gracefully and return strings.
Uses lazy imports to avoid triggering config loading at module level.
"""

from __future__ import annotations


class TestBillingAgent:
    def test_error_returns_fallback_message(self, monkeypatch):
        """Billing agent should return a user-friendly message on error."""
        import agents.billing_agent as mod

        def _raise(*args, **kwargs):
            raise Exception("API down")

        monkeypatch.setattr(mod, "_get_client", _raise)

        from agents.billing_agent import answer_billing_query
        result = answer_billing_query("test")
        assert isinstance(result, str)
        assert "error" in result.lower() or "sorry" in result.lower()

    def test_accepts_chat_history(self, monkeypatch):
        """Billing agent should accept optional chat_history parameter."""
        import agents.billing_agent as mod

        monkeypatch.setattr(mod, "_get_client", lambda: (_ for _ in ()).throw(Exception("skip")))

        from agents.billing_agent import answer_billing_query
        # Should not TypeError on extra argument
        result = answer_billing_query("test", chat_history="User: prev\nAssistant: ans")
        assert isinstance(result, str)


class TestTechAgent:
    def test_error_returns_fallback_message(self, monkeypatch):
        import agents.tech_agent as mod

        monkeypatch.setattr(mod, "_get_client", lambda: (_ for _ in ()).throw(Exception("skip")))

        from agents.tech_agent import answer_tech_query
        result = answer_tech_query("test")
        assert isinstance(result, str)
        assert "error" in result.lower() or "sorry" in result.lower()


class TestGeneralAgent:
    def test_error_returns_fallback_message(self, monkeypatch):
        import agents.general_agent as mod

        monkeypatch.setattr(mod, "_get_client", lambda: (_ for _ in ()).throw(Exception("skip")))

        from agents.general_agent import answer_general_query
        result = answer_general_query("test")
        assert isinstance(result, str)
        assert "error" in result.lower() or "sorry" in result.lower()
