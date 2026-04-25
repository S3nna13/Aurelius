"""Tests for multiagent weighted consensus."""
from __future__ import annotations

import pytest

from src.multiagent.consensus import Vote, WeightedConsensus, ConsensusResult


class TestWeightedConsensus:
    """Test WeightedConsensus voting."""
    
    def test_simple_majority(self):
        """Test simple majority wins."""
        votes = [
            Vote("agent1", "A", 1.0),
            Vote("agent2", "B", 1.0),
            Vote("agent3", "A", 1.0),
        ]
        
        result = WeightedConsensus().vote(votes)
        
        assert result.winner == "A"
        assert result.votes_for["A"] == 2.0
        assert result.quorum_met is True
    
    def test_weighted_voting(self):
        """Test weighted voting."""
        votes = [
            Vote("agent1", "A", 1.0),
            Vote("agent2", "B", 3.0),
        ]
        
        result = WeightedConsensus().vote(votes)
        
        assert result.winner == "B"
        assert result.votes_for["B"] == 3.0
    
    def test_no_quorum(self):
        """Test no quorum when sparse voting."""
        votes = [
            Vote("agent1", "A", 0.3),
            Vote("agent2", "B", 0.3),
        ]
        
        result = WeightedConsensus(quorum_threshold=0.8).vote(votes)
        
        assert result.quorum_met is False
        assert result.runoff is True
    
    def test_empty_votes(self):
        """Test empty votes."""
        result = WeightedConsensus().vote([])
        
        assert result.winner is None
        assert result.total_votes == 0.0