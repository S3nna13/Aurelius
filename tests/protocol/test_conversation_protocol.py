"""Tests for src/protocol/conversation_protocol.py (~50 tests)."""
import pytest
from src.protocol.conversation_protocol import (
    ConversationProtocol,
    ConversationState,
    TurnValidationError,
    CONVERSATION_PROTOCOL,
)


class TestConversationStateEnum:
    def test_enum_count(self):
        assert len(ConversationState) == 6

    def test_init_value(self):
        assert ConversationState.INIT == "init"

    def test_user_turn_value(self):
        assert ConversationState.USER_TURN == "user_turn"

    def test_assistant_turn_value(self):
        assert ConversationState.ASSISTANT_TURN == "assistant_turn"

    def test_tool_call_value(self):
        assert ConversationState.TOOL_CALL == "tool_call"

    def test_tool_result_value(self):
        assert ConversationState.TOOL_RESULT == "tool_result"

    def test_ended_value(self):
        assert ConversationState.ENDED == "ended"

    def test_is_str_subclass(self):
        assert isinstance(ConversationState.INIT, str)


class TestConversationProtocol:
    def setup_method(self):
        self.proto = ConversationProtocol()

    # Initial state
    def test_initial_state_is_init(self):
        assert self.proto.state == ConversationState.INIT

    # INIT → USER_TURN
    def test_transition_user_from_init_returns_user_turn(self):
        result = self.proto.transition("user")
        assert result == ConversationState.USER_TURN

    def test_transition_user_from_init_updates_state(self):
        self.proto.transition("user")
        assert self.proto.state == ConversationState.USER_TURN

    def test_transition_invalid_role_from_init_raises(self):
        with pytest.raises(TurnValidationError):
            self.proto.transition("assistant")

    def test_transition_tool_call_from_init_raises(self):
        with pytest.raises(TurnValidationError):
            self.proto.transition("tool_call")

    def test_transition_end_from_init_raises(self):
        with pytest.raises(TurnValidationError):
            self.proto.transition("end")

    # USER_TURN → ASSISTANT_TURN
    def test_transition_assistant_from_user_turn(self):
        self.proto.transition("user")
        result = self.proto.transition("assistant")
        assert result == ConversationState.ASSISTANT_TURN

    def test_transition_invalid_from_user_turn_raises(self):
        self.proto.transition("user")
        with pytest.raises(TurnValidationError):
            self.proto.transition("tool_result")

    def test_transition_user_from_user_turn_raises(self):
        self.proto.transition("user")
        with pytest.raises(TurnValidationError):
            self.proto.transition("user")

    # ASSISTANT_TURN branches
    def test_transition_user_from_assistant_turn(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        result = self.proto.transition("user")
        assert result == ConversationState.USER_TURN

    def test_transition_tool_call_from_assistant_turn(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        result = self.proto.transition("tool_call")
        assert result == ConversationState.TOOL_CALL

    def test_transition_end_from_assistant_turn(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        result = self.proto.transition("end")
        assert result == ConversationState.ENDED

    def test_transition_invalid_from_assistant_turn_raises(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        with pytest.raises(TurnValidationError):
            self.proto.transition("tool_result")

    # TOOL_CALL → TOOL_RESULT
    def test_transition_tool_result_from_tool_call(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("tool_call")
        result = self.proto.transition("tool_result")
        assert result == ConversationState.TOOL_RESULT

    def test_transition_invalid_from_tool_call_raises(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("tool_call")
        with pytest.raises(TurnValidationError):
            self.proto.transition("assistant")

    def test_transition_user_from_tool_call_raises(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("tool_call")
        with pytest.raises(TurnValidationError):
            self.proto.transition("user")

    # TOOL_RESULT → ASSISTANT_TURN
    def test_transition_assistant_from_tool_result(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("tool_call")
        self.proto.transition("tool_result")
        result = self.proto.transition("assistant")
        assert result == ConversationState.ASSISTANT_TURN

    def test_transition_invalid_from_tool_result_raises(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("tool_call")
        self.proto.transition("tool_result")
        with pytest.raises(TurnValidationError):
            self.proto.transition("user")

    # ENDED
    def test_transition_from_ended_raises(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("end")
        with pytest.raises(TurnValidationError):
            self.proto.transition("user")

    def test_transition_from_ended_error_message(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("end")
        with pytest.raises(TurnValidationError, match="ended"):
            self.proto.transition("anything")

    # reset()
    def test_reset_returns_to_init(self):
        self.proto.transition("user")
        self.proto.reset()
        assert self.proto.state == ConversationState.INIT

    def test_reset_clears_history(self):
        self.proto.transition("user")
        self.proto.reset()
        assert self.proto.history() == []

    def test_reset_allows_transitions_again(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("end")
        self.proto.reset()
        result = self.proto.transition("user")
        assert result == ConversationState.USER_TURN

    # valid_next_roles()
    def test_valid_next_roles_from_init(self):
        roles = self.proto.valid_next_roles()
        assert "user" in roles

    def test_valid_next_roles_nonempty_from_user_turn(self):
        self.proto.transition("user")
        assert len(self.proto.valid_next_roles()) > 0

    def test_valid_next_roles_nonempty_from_assistant_turn(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        assert len(self.proto.valid_next_roles()) > 0

    def test_valid_next_roles_from_assistant_turn_has_three(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        roles = self.proto.valid_next_roles()
        assert len(roles) == 3

    def test_valid_next_roles_from_ended_is_empty(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        self.proto.transition("end")
        assert self.proto.valid_next_roles() == []

    def test_valid_next_roles_returns_list(self):
        assert isinstance(self.proto.valid_next_roles(), list)

    # history()
    def test_history_empty_initially(self):
        assert self.proto.history() == []

    def test_history_records_single_transition(self):
        self.proto.transition("user")
        h = self.proto.history()
        assert len(h) == 1
        assert h[0] == ("user", ConversationState.USER_TURN)

    def test_history_records_multiple_transitions(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        h = self.proto.history()
        assert len(h) == 2

    def test_history_records_role_and_resulting_state(self):
        self.proto.transition("user")
        self.proto.transition("assistant")
        h = self.proto.history()
        assert h[1] == ("assistant", ConversationState.ASSISTANT_TURN)

    def test_history_returns_list_of_tuples(self):
        self.proto.transition("user")
        h = self.proto.history()
        assert isinstance(h, list)
        assert isinstance(h[0], tuple)

    def test_history_is_copy_not_reference(self):
        self.proto.transition("user")
        h1 = self.proto.history()
        self.proto.transition("assistant")
        h2 = self.proto.history()
        assert len(h1) == 1
        assert len(h2) == 2

    # Singleton
    def test_conversation_protocol_singleton_exists(self):
        assert CONVERSATION_PROTOCOL is not None
        assert isinstance(CONVERSATION_PROTOCOL, ConversationProtocol)
