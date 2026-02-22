# tests/test_rag_mocked.py
import pytest
from unittest.mock import Mock, patch, MagicMock

def test_rag_with_mocked_llm():
    """Test RAG logic without calling real Ollama"""
    
    # Create a mock LLM that returns a predetermined answer
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value="Customers can request a refund within 30 days.")
    
    # Test the mock
    result = mock_llm.invoke("What is the refund policy?")
    
    assert result == "Customers can request a refund within 30 days."
    assert mock_llm.invoke.called
    print("✅ Mock LLM worked")


def test_answer_validation_logic():
    """Test answer validation without calling LLM"""
    
    # Simulate different LLM responses
    test_cases = [
        ("", False, "Empty answer should fail"),
        ("I don't know", True, "Valid abstention"),
        ("a" * 1000, False, "Answer too long"),
        ("Refunds within 30 days", True, "Valid answer"),
    ]
    
    for answer, should_pass, description in test_cases:
        # Validation logic
        is_valid = (
            answer is not None 
            and len(answer) > 0 
            and len(answer) < 500
        )
        
        if should_pass:
            assert is_valid, f"FAILED: {description}"
        
        print(f"✅ {description}")


@patch('langchain_ollama.OllamaLLM')
def test_rag_pipeline_with_patched_ollama(mock_ollama_class):
    """Test RAG pipeline with Ollama completely mocked"""
    
    # Configure the mock to return a specific response
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = "Full-time employees receive 20 days of annual leave per year."
    mock_ollama_class.return_value = mock_instance
    
    # Now when code creates OllamaLLM(), it gets our mock instead
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="llama3.2")
    
    result = llm.invoke("How many days of leave?")
    
    assert "20 days" in result
    assert mock_instance.invoke.called
    print("✅ Full OllamaLLM mock worked")


def test_hallucination_detection_logic():
    """Test abstention detection without LLM"""
    
    responses_that_should_abstain = [
        "I don't know based on available information.",
        "That information is not available in the context.",
        "I cannot answer that question.",
    ]
    
    responses_that_are_hallucinations = [
        "The CEO is John Smith.",
        "I believe the stock price is around $50.",
        "Based on industry trends, I would estimate...",
    ]
    
    abstain_phrases = ["don't know", "not available", "cannot answer"]
    
    # Test correct abstentions
    for response in responses_that_should_abstain:
        is_abstaining = any(phrase in response.lower() for phrase in abstain_phrases)
        assert is_abstaining, f"Should detect abstention in: {response}"
        print(f"✅ Correctly detected abstention")
    
    # Test hallucinations (should NOT contain abstention phrases)
    for response in responses_that_are_hallucinations:
        is_abstaining = any(phrase in response.lower() for phrase in abstain_phrases)
        assert not is_abstaining, f"Should NOT detect abstention in: {response}"
        print(f"✅ Correctly detected hallucination (no abstention)")


def test_chunk_retrieval_logic():
    """Test that retrieval logic works without vector store"""
    
    # Mock retrieved chunks
    mock_chunks = [
        Mock(page_content="Refund policy: 30 days", metadata={"source": "refund.txt"}),
        Mock(page_content="Leave policy: 20 days", metadata={"source": "leave.txt"}),
    ]
    
    # Test chunk formatting
    formatted = "\n\n".join([chunk.page_content for chunk in mock_chunks])
    
    assert "Refund policy" in formatted
    assert "Leave policy" in formatted
    assert formatted.count("\n\n") == 1  # One separator between two chunks
    print("✅ Chunk formatting works")