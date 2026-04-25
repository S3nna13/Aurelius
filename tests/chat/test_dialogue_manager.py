import pytest

from src.chat.dialogue_manager import DialogueManager, DialogueState, Turn


def make_dm(slots=None):
    return DialogueManager(slots=slots)


def test_create_dialogue_returns_context():
    dm = make_dm()
    ctx = dm.create_dialogue()
    assert ctx.dialogue_id
    assert ctx.turns == []
    assert ctx.current_state == DialogueState.GREETING


def test_create_dialogue_with_goal():
    dm = make_dm()
    ctx = dm.create_dialogue(goal="book flight")
    assert ctx.goal == "book flight"


def test_list_dialogues_empty():
    dm = make_dm()
    assert dm.list_dialogues() == []


def test_list_dialogues_after_create():
    dm = make_dm()
    ctx = dm.create_dialogue()
    assert ctx.dialogue_id in dm.list_dialogues()


def test_add_turn_appended():
    dm = make_dm()
    ctx = dm.create_dialogue()
    turn = dm.add_turn(ctx.dialogue_id, "user", "Hello there")
    assert len(ctx.turns) == 1
    assert turn.content == "Hello there"
    assert turn.role == "user"
    assert turn.turn_id == 0


def test_add_multiple_turns_increment_ids():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "Hi")
    dm.add_turn(ctx.dialogue_id, "assistant", "Hello")
    assert ctx.turns[0].turn_id == 0
    assert ctx.turns[1].turn_id == 1


def test_get_context_returns_same():
    dm = make_dm()
    ctx = dm.create_dialogue()
    assert dm.get_context(ctx.dialogue_id) is ctx


def test_get_context_missing_returns_none():
    dm = make_dm()
    assert dm.get_context("nonexistent") is None


def test_get_history_all():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "a")
    dm.add_turn(ctx.dialogue_id, "assistant", "b")
    history = dm.get_history(ctx.dialogue_id)
    assert len(history) == 2


def test_get_history_last_n():
    dm = make_dm()
    ctx = dm.create_dialogue()
    for i in range(5):
        dm.add_turn(ctx.dialogue_id, "user", str(i))
    history = dm.get_history(ctx.dialogue_id, last_n=2)
    assert len(history) == 2
    assert history[-1].content == "4"


def test_get_history_missing_dialogue():
    dm = make_dm()
    assert dm.get_history("bad_id") == []


def test_fill_slot_directly():
    dm = make_dm(slots=["destination"])
    ctx = dm.create_dialogue()
    dm.fill_slot(ctx.dialogue_id, "destination", "Paris")
    assert ctx.slot_values["destination"] == "Paris"


def test_slots_filled_from_entities():
    dm = make_dm(slots=["destination", "date"])
    ctx = dm.create_dialogue()
    dm.add_turn(
        ctx.dialogue_id,
        "user",
        "I want to go to Paris on 12/25",
        entities={"destination": "Paris", "date": "12/25"},
    )
    assert ctx.slot_values["destination"] == "Paris"
    assert ctx.slot_values["date"] == "12/25"


def test_state_greeting_on_first_turn():
    dm = make_dm()
    ctx = dm.create_dialogue()
    state = dm.update_state(ctx)
    assert state == DialogueState.GREETING


def test_state_closing_on_bye():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "bye")
    assert ctx.current_state == DialogueState.CLOSING


def test_state_closing_on_thank_you():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "thank you so much")
    assert ctx.current_state == DialogueState.CLOSING


def test_state_clarification():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "what do you mean by that?")
    assert ctx.current_state == DialogueState.CLARIFICATION


def test_state_confirmation_when_all_slots_filled():
    dm = make_dm(slots=["dest"])
    ctx = dm.create_dialogue()
    dm.fill_slot(ctx.dialogue_id, "dest", "London")
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "sounds good")
    assert ctx.current_state == DialogueState.CONFIRMATION


def test_state_information_gathering_with_no_slots_filled():
    dm = make_dm(slots=["dest", "date", "pax"])
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "I need a flight")
    assert ctx.current_state == DialogueState.INFORMATION_GATHERING


def test_state_task_execution_half_slots_filled():
    dm = make_dm(slots=["dest", "date"])
    ctx = dm.create_dialogue()
    dm.fill_slot(ctx.dialogue_id, "dest", "Rome")
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "searching...")
    assert ctx.current_state == DialogueState.TASK_EXECUTION


def test_is_complete_false_initially():
    dm = make_dm(slots=["dest"])
    ctx = dm.create_dialogue()
    assert not dm.is_complete(ctx.dialogue_id)


def test_is_complete_true_when_slots_filled():
    dm = make_dm(slots=["dest"])
    ctx = dm.create_dialogue()
    dm.fill_slot(ctx.dialogue_id, "dest", "Tokyo")
    assert dm.is_complete(ctx.dialogue_id)


def test_is_complete_true_when_closing():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "bye")
    assert dm.is_complete(ctx.dialogue_id)


def test_is_complete_missing_dialogue():
    dm = make_dm()
    assert not dm.is_complete("nope")


def test_turn_has_timestamp():
    dm = make_dm()
    ctx = dm.create_dialogue()
    turn = dm.add_turn(ctx.dialogue_id, "user", "hi")
    assert turn.timestamp > 0


def test_no_slots_state_task_execution():
    dm = make_dm()
    ctx = dm.create_dialogue()
    dm.add_turn(ctx.dialogue_id, "user", "hello")
    dm.add_turn(ctx.dialogue_id, "user", "do something for me")
    assert ctx.current_state == DialogueState.TASK_EXECUTION


def test_intent_stored_on_turn():
    dm = make_dm()
    ctx = dm.create_dialogue()
    turn = dm.add_turn(ctx.dialogue_id, "user", "hi", intent="greeting")
    assert turn.intent == "greeting"


def test_multiple_dialogues_isolated():
    dm = make_dm(slots=["x"])
    ctx1 = dm.create_dialogue()
    ctx2 = dm.create_dialogue()
    dm.fill_slot(ctx1.dialogue_id, "x", "val")
    assert "x" not in ctx2.slot_values
