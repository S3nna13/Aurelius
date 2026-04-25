"""Tests for multi_agent_conversation.py"""
import pytest
from src.chat.multi_agent_conversation import (
    AgentRole,
    AgentProfile,
    ConversationMessage,
    MultiAgentConversation,
)


# --- AgentRole enum ---

def test_agent_role_orchestrator():
    assert AgentRole.ORCHESTRATOR == "orchestrator"

def test_agent_role_worker():
    assert AgentRole.WORKER == "worker"

def test_agent_role_critic():
    assert AgentRole.CRITIC == "critic"

def test_agent_role_summarizer():
    assert AgentRole.SUMMARIZER == "summarizer"

def test_agent_role_user_proxy():
    assert AgentRole.USER_PROXY == "user_proxy"

def test_agent_role_count():
    assert len(AgentRole) == 5

def test_agent_role_is_str():
    assert isinstance(AgentRole.WORKER, str)


# --- AgentProfile ---

def test_agent_profile_auto_id():
    p = AgentProfile(name="Alice", role=AgentRole.WORKER)
    assert p.agent_id is not None
    assert len(p.agent_id) == 8

def test_agent_profile_unique_ids():
    p1 = AgentProfile(name="A", role=AgentRole.WORKER)
    p2 = AgentProfile(name="B", role=AgentRole.WORKER)
    assert p1.agent_id != p2.agent_id

def test_agent_profile_name():
    p = AgentProfile(name="Bob", role=AgentRole.CRITIC)
    assert p.name == "Bob"

def test_agent_profile_role():
    p = AgentProfile(name="Bob", role=AgentRole.CRITIC)
    assert p.role == AgentRole.CRITIC

def test_agent_profile_default_system_prompt():
    p = AgentProfile(name="C", role=AgentRole.ORCHESTRATOR)
    assert p.system_prompt == ""

def test_agent_profile_custom_system_prompt():
    p = AgentProfile(name="C", role=AgentRole.ORCHESTRATOR, system_prompt="You are helpful.")
    assert p.system_prompt == "You are helpful."

def test_agent_profile_custom_id():
    p = AgentProfile(name="D", role=AgentRole.WORKER, agent_id="abc12345")
    assert p.agent_id == "abc12345"


# --- ConversationMessage ---

def test_message_auto_id():
    m = ConversationMessage(from_agent="a", to_agent=None, content="hello")
    assert m.id is not None
    assert len(m.id) == 8

def test_message_unique_ids():
    m1 = ConversationMessage(from_agent="a", to_agent=None, content="x")
    m2 = ConversationMessage(from_agent="a", to_agent=None, content="y")
    assert m1.id != m2.id

def test_message_timestamp_not_empty():
    m = ConversationMessage(from_agent="a", to_agent=None, content="hi")
    assert m.timestamp != ""

def test_message_default_thread_id():
    m = ConversationMessage(from_agent="a", to_agent=None, content="hi")
    assert m.thread_id == "main"

def test_message_custom_thread_id():
    m = ConversationMessage(from_agent="a", to_agent=None, content="hi", thread_id="side")
    assert m.thread_id == "side"

def test_message_to_agent_none():
    m = ConversationMessage(from_agent="a", to_agent=None, content="broadcast")
    assert m.to_agent is None

def test_message_to_agent_set():
    m = ConversationMessage(from_agent="a", to_agent="b", content="direct")
    assert m.to_agent == "b"


# --- MultiAgentConversation ---

def test_mac_default_max_turns():
    mac = MultiAgentConversation()
    assert mac.max_turns == 50

def test_mac_custom_max_turns():
    mac = MultiAgentConversation(max_turns=10)
    assert mac.max_turns == 10

def test_mac_add_agent_and_agents_roundtrip():
    mac = MultiAgentConversation()
    p = AgentProfile(name="Alice", role=AgentRole.WORKER)
    mac.add_agent(p)
    assert p in mac.agents()

def test_mac_agents_empty_initially():
    mac = MultiAgentConversation()
    assert mac.agents() == []

def test_mac_agents_multiple():
    mac = MultiAgentConversation()
    p1 = AgentProfile(name="A", role=AgentRole.WORKER)
    p2 = AgentProfile(name="B", role=AgentRole.CRITIC)
    mac.add_agent(p1)
    mac.add_agent(p2)
    assert len(mac.agents()) == 2

def test_mac_send_returns_message():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "hello")
    assert isinstance(msg, ConversationMessage)

def test_mac_send_content():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "hello world")
    assert msg.content == "hello world"

def test_mac_send_from_agent():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "hi")
    assert msg.from_agent == "agent1"

def test_mac_send_to_agent_default_none():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "broadcast")
    assert msg.to_agent is None

def test_mac_send_to_agent_set():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "direct", to_id="agent2")
    assert msg.to_agent == "agent2"

def test_mac_send_default_thread():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "hi")
    assert msg.thread_id == "main"

def test_mac_send_custom_thread():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "hi", thread_id="review")
    assert msg.thread_id == "review"

def test_mac_send_appears_in_messages():
    mac = MultiAgentConversation()
    msg = mac.send("agent1", "hello")
    assert msg in mac.messages()

def test_mac_messages_filter_thread_id():
    mac = MultiAgentConversation()
    mac.send("a", "main msg", thread_id="main")
    mac.send("b", "side msg", thread_id="side")
    main_msgs = mac.messages(thread_id="main")
    assert all(m.thread_id == "main" for m in main_msgs)
    assert len(main_msgs) == 1

def test_mac_messages_filter_thread_excludes_other():
    mac = MultiAgentConversation()
    mac.send("a", "main msg", thread_id="main")
    mac.send("b", "side msg", thread_id="side")
    side_msgs = mac.messages(thread_id="side")
    assert len(side_msgs) == 1
    assert side_msgs[0].content == "side msg"

def test_mac_messages_filter_agent_from():
    mac = MultiAgentConversation()
    mac.send("alice", "hello", to_id="bob")
    mac.send("carol", "hi", to_id="dave")
    result = mac.messages(agent_id="alice")
    assert len(result) == 1
    assert result[0].from_agent == "alice"

def test_mac_messages_filter_agent_to():
    mac = MultiAgentConversation()
    mac.send("alice", "hello", to_id="bob")
    mac.send("carol", "hi", to_id="dave")
    result = mac.messages(agent_id="bob")
    assert len(result) == 1
    assert result[0].to_agent == "bob"

def test_mac_messages_filter_agent_from_or_to():
    mac = MultiAgentConversation()
    mac.send("alice", "hello", to_id="bob")
    mac.send("bob", "reply", to_id="alice")
    result = mac.messages(agent_id="bob")
    assert len(result) == 2

def test_mac_messages_no_filter():
    mac = MultiAgentConversation()
    mac.send("a", "one")
    mac.send("b", "two")
    assert len(mac.messages()) == 2

def test_mac_turn_count_zero_initially():
    mac = MultiAgentConversation()
    assert mac.turn_count() == 0

def test_mac_turn_count_increments():
    mac = MultiAgentConversation()
    mac.send("a", "one")
    assert mac.turn_count() == 1

def test_mac_turn_count_multiple():
    mac = MultiAgentConversation()
    mac.send("a", "one")
    mac.send("b", "two")
    mac.send("c", "three")
    assert mac.turn_count() == 3

def test_mac_thread_ids_includes_main():
    mac = MultiAgentConversation()
    mac.send("a", "hello")
    assert "main" in mac.thread_ids()

def test_mac_thread_ids_unique():
    mac = MultiAgentConversation()
    mac.send("a", "m1", thread_id="main")
    mac.send("b", "m2", thread_id="main")
    mac.send("c", "m3", thread_id="side")
    ids = mac.thread_ids()
    assert len(ids) == len(set(ids))

def test_mac_thread_ids_multiple():
    mac = MultiAgentConversation()
    mac.send("a", "m1", thread_id="main")
    mac.send("b", "m2", thread_id="side")
    mac.send("c", "m3", thread_id="other")
    ids = mac.thread_ids()
    assert set(ids) == {"main", "side", "other"}

def test_mac_thread_ids_empty_initially():
    mac = MultiAgentConversation()
    assert mac.thread_ids() == []

def test_mac_messages_combined_filter():
    mac = MultiAgentConversation()
    mac.send("alice", "hello", to_id="bob", thread_id="main")
    mac.send("alice", "side note", to_id="bob", thread_id="side")
    result = mac.messages(thread_id="main", agent_id="alice")
    assert len(result) == 1
    assert result[0].thread_id == "main"
    assert result[0].from_agent == "alice"
