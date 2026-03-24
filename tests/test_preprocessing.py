from src.asrs_sum.core.preprocessing import clean_text, safe_truncate_words


def test_clean_text_removes_extra_spaces():
    text = "Hello   world \n this  is   a test ."
    cleaned = clean_text(text)
    assert "  " not in cleaned
    assert "\n" not in cleaned


def test_safe_truncate_words():
    text = "one two three four five"
    assert safe_truncate_words(text, 3) == "one two three"