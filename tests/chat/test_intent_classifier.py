import pytest

from src.chat.intent_classifier import Intent, IntentClassifier, IntentResult


@pytest.fixture
def clf():
    return IntentClassifier()


def test_classify_greeting_hello(clf):
    result = clf.classify("Hello there!")
    assert result.intent == Intent.GREETING


def test_classify_greeting_hi(clf):
    result = clf.classify("hi, how are you")
    assert result.intent == Intent.GREETING


def test_classify_farewell_bye(clf):
    result = clf.classify("bye for now")
    assert result.intent == Intent.FAREWELL


def test_classify_farewell_thank_you(clf):
    result = clf.classify("thank you very much")
    assert result.intent == Intent.FAREWELL


def test_classify_question_wh_word(clf):
    result = clf.classify("What time does it start?")
    assert result.intent == Intent.QUESTION


def test_classify_question_mark(clf):
    result = clf.classify("Is this working?")
    assert result.intent == Intent.QUESTION


def test_classify_request_please(clf):
    result = clf.classify("Please send me the report")
    assert result.intent == Intent.REQUEST


def test_classify_request_can_you(clf):
    result = clf.classify("Can you help me with this?")
    assert result.intent in {Intent.REQUEST, Intent.QUESTION}


def test_classify_complaint_not_working(clf):
    result = clf.classify("This is not working at all")
    assert result.intent == Intent.COMPLAINT


def test_classify_complaint_problem(clf):
    result = clf.classify("There is a problem with the upload")
    assert result.intent == Intent.COMPLAINT


def test_classify_confirmation_yes(clf):
    result = clf.classify("yes that is correct")
    assert result.intent == Intent.CONFIRMATION


def test_classify_denial_no(clf):
    result = clf.classify("no that is not right")
    assert result.intent == Intent.DENIAL


def test_classify_clarification_request(clf):
    result = clf.classify("what do you mean by that?")
    assert result.intent == Intent.CLARIFICATION_REQUEST


def test_classify_clarification_explain(clf):
    result = clf.classify("could you explain that again?")
    assert result.intent == Intent.CLARIFICATION_REQUEST


def test_classify_unknown_empty(clf):
    result = clf.classify("")
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0


def test_confidence_between_zero_and_one(clf):
    result = clf.classify("Hello, please help me!")
    assert 0.0 <= result.confidence <= 1.0


def test_keywords_matched_non_empty(clf):
    result = clf.classify("hello there")
    assert len(result.keywords_matched) > 0


def test_extract_entities_numbers(clf):
    entities = clf.extract_entities("I need 3 tickets and 42 seats")
    assert "numbers" in entities
    assert "3" in entities["numbers"]
    assert "42" in entities["numbers"]


def test_extract_entities_dates(clf):
    entities = clf.extract_entities("Book for 12/25/2024")
    assert "dates" in entities
    assert "12/25/2024" in entities["dates"]


def test_extract_entities_quoted(clf):
    entities = clf.extract_entities('My name is "John Doe"')
    assert "quoted" in entities
    assert "John Doe" in entities["quoted"]


def test_extract_entities_empty_text(clf):
    entities = clf.extract_entities("")
    assert entities == {}


def test_extract_entities_no_match(clf):
    entities = clf.extract_entities("Hello world")
    assert "numbers" not in entities
    assert "dates" not in entities
    assert "quoted" not in entities


def test_batch_classify_returns_list(clf):
    texts = ["hello", "bye", "what time is it?"]
    results = clf.batch_classify(texts)
    assert len(results) == 3
    assert all(isinstance(r, IntentResult) for r in results)


def test_batch_classify_correct_intents(clf):
    results = clf.batch_classify(["hello", "bye"])
    assert results[0].intent == Intent.GREETING
    assert results[1].intent == Intent.FAREWELL


def test_add_pattern_custom(clf):
    clf.add_pattern(Intent.COMPLAINT, r"\bterrible\b")
    result = clf.classify("This is terrible service")
    assert result.intent == Intent.COMPLAINT


def test_custom_patterns_at_init():
    custom = {Intent.REQUEST: [r"\bfetch\b"]}
    clf = IntentClassifier(custom_patterns=custom)
    result = clf.classify("fetch the data")
    assert result.intent == Intent.REQUEST


def test_classify_entities_in_result(clf):
    result = clf.classify('I need 5 tickets for "rock concert"')
    assert "numbers" in result.entities
    assert "quoted" in result.entities


def test_date_entity_not_counted_as_number(clf):
    entities = clf.extract_entities("Schedule for 03/15/2025")
    assert "dates" in entities
    if "numbers" in entities:
        assert "03" not in entities["numbers"]
        assert "15" not in entities["numbers"]
