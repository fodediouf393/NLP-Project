from src.asrs_sum.core.topk_summarizer import TopKExtractiveSummarizer


def test_summarizer_returns_non_empty_summary():
    text = (
        "During taxi the aircraft entered the wrong runway due to poor visibility. "
        "ATC instructed the crew to stop immediately. "
        "The captain corrected the aircraft position and returned to the assigned taxiway."
    )

    summarizer = TopKExtractiveSummarizer(top_k_sentences=2, max_summary_words=40)
    result = summarizer.summarize(text)

    assert result.summary
    assert len(result.selected_sentences) >= 1


def test_summarizer_handles_empty_text():
    summarizer = TopKExtractiveSummarizer()
    result = summarizer.summarize("")
    assert result.summary == ""
    assert result.selected_sentences == []