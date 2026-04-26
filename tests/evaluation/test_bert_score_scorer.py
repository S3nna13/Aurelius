"""Tests for bert_score_scorer."""
from __future__ import annotations
import pytest
from src.evaluation.bert_score_scorer import BertScoreScorer

class TestBertScoreScorer:
    def test_exact_match(self): s=BertScoreScorer(); r=s.score("hello world","hello world"); assert r.precision>0.9
    def test_no_match(self): s=BertScoreScorer(); r=s.score("hello","xyz"); assert r.recall<0.5
    def test_empty(self): s=BertScoreScorer(); r=s.score("",""); assert r.f1==0.0
    def test_partial(self): s=BertScoreScorer(); r=s.score("cat sat","cat slept"); assert 0<r.f1<1.0
