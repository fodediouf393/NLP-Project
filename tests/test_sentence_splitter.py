from src.asrs_sum.core.sentence_splitter import split_into_sentences


def test_split_into_sentences_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].index == 0
    assert sentences[1].index == 1
    assert sentences[2].index == 2