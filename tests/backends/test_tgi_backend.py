"""Tests for tgi_backend."""
from __future__ import annotations
from src.backends.tgi_backend import TgiBackend

class TestTgiBackend:
    def test_name(self): b=TgiBackend();assert b.name=="tgi"
    def test_health_initially_false(self): b=TgiBackend();assert b.health()==False
    def test_configure_sets_url(self): b=TgiBackend();b.configure(endpoint="http://localhost:8080");assert b._endpoint=="http://localhost:8080"
    def test_capabilities(self): c=TgiBackend().capabilities();assert "text_generation" in c
