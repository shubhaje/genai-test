# tests/test_rag_basic.py
import pytest

def test_rag_pipeline_exists(rag_pipeline):
    """Test that RAG pipeline fixture loads successfully"""
    assert rag_pipeline is not None
    print("✅ Pipeline exists")


def test_answer_not_empty(rag_pipeline):
    """Test that RAG returns non-empty responses"""
    question = "What is the refund policy?"
    answer = rag_pipeline.invoke(question)
    
    assert answer is not None
    assert len(answer) > 0
    assert isinstance(answer, str)
    print(f"✅ Got answer: {answer[:80]}...")


def test_answer_reasonable_length(rag_pipeline):
    """Test that answers are neither too short nor too long"""
    question = "How many days of annual leave do employees get?"
    answer = rag_pipeline.invoke(question)
    
    # Should be more than a few words but less than essay-length
    assert len(answer) > 20, "Answer too short"
    assert len(answer) < 500, "Answer too long (likely hallucination)"
    print(f"✅ Answer length OK: {len(answer)} chars")


def test_answerable_questions_get_answers(rag_pipeline, sample_questions):
    """Test that questions with answers in docs don't return 'I don't know'"""
    for question in sample_questions['answerable']:
        answer = rag_pipeline.invoke(question)
        
        # Should NOT abstain on answerable questions
        assert "don't know" not in answer.lower(), f"Pipeline incorrectly abstained on: {question}"
        print(f"✅ {question[:50]}... → Answered (not abstained)")


def test_unanswerable_questions_abstain(rag_pipeline, sample_questions):
    """Test that questions without answers properly abstain"""
    for question in sample_questions['unanswerable']:
        answer = rag_pipeline.invoke(question)
        
        # SHOULD abstain on unanswerable questions
        abstain_phrases = ["don't know", "not available", "no information"]
        assert any(phrase in answer.lower() for phrase in abstain_phrases), \
            f"Pipeline should have abstained but didn't: {question}\nGot: {answer}"
        print(f"✅ {question[:50]}... → Abstained correctly")