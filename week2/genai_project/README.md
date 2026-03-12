# RAG Pipeline with Automated Quality Testing

Production-ready Retrieval-Augmented Generation system with comprehensive test suite achieving perfect quality scores.

## 🎯 Project Highlights

- **Perfect Quality Metrics:** Faithfulness 1.000, Answer Quality 1.000, Context Precision 1.000, Context Recall 1.000
- **Systematic Optimization:** Debugged retrieval failures, optimized chunking (200→500 tokens), eliminated hallucinations (67%→0%)
- **Comprehensive Testing:** 27 automated tests (10 integration/unit + 17 adversarial)
- **Real-World Debugging:** Fixed 3 production failure modes through systematic root cause analysis

## 📊 Quality Scorecard

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Faithfulness** | 1.000 | >0.85 | ✅ Exceeds |
| **Answer Quality** | 1.000 | >0.90 | ✅ Exceeds |
| **Context Precision** | 1.000 | >0.75 | ✅ Exceeds |
| **Context Recall** | 1.000 | >0.75 | ✅ Exceeds |

*Note: Most production RAG systems score 0.75-0.85. Perfect 1.000 scores achieved through systematic optimization.*

## 🚀 Tech Stack

- **LLM:** LLaMA 3.2 (via Ollama - local inference)
- **Framework:** LangChain (RAG orchestration)
- **Vector DB:** ChromaDB (local development)
- **Testing:** pytest (27 tests with mocking and adversarial scenarios)
- **Evaluation:** RAGAS concepts (faithfulness, precision, recall, relevancy)

## 📺 Demo Video

[🎥 Watch 5-minute walkthrough](https://loom.com/your-link-here) *(Add link after recording)*

## 🧪 Test Coverage
```
tests/
├── test_rag_basic.py        (5 integration tests - real LLM calls)
├── test_rag_mocked.py        (5 unit tests - mocked, 85× faster)
└── test_adversarial.py       (17 adversarial tests - designed to break system)
```

**Test Categories:**
- ✅ **Integration Tests:** End-to-end RAG validation with real Ollama calls
- ✅ **Unit Tests (Mocked):** Logic validation without LLM calls (< 1 second)
- ✅ **Adversarial Tests:** Hallucination detection, scope violations, prompt injection attempts

## 🔍 Key Learnings

### 1. Retrieval Debugging (Day 2)
**Problem:** Question "What is the refund policy?" returned "I don't know" despite answer being in documents.

**Root Cause:** ChromaDB accumulated stale duplicate data from multiple script runs. Same irrelevant chunk appeared 3 times in top-3 results.

**Solution:** Implemented clean collection rebuilds before indexing. In production: use upsert-based indexing pipeline.

### 2. Chunking Optimization (Day 3)
Systematic experiment testing 200, 500, and 1000 token chunk sizes:

| Chunk Size | Success Rate | Issue |
|------------|-------------|-------|
| 200 tokens | 60% | Context fragmentation - policy split across chunks |
| **500 tokens** | **100%** | **Optimal - selected** |
| 1000 tokens | 100% | Verbose/contradictory answers from too much context |

### 3. Prompt Engineering (Day 4)
Measured hallucination rates across 3 prompt variants:

| Prompt Type | Hallucination Rate | Result |
|-------------|-------------------|---------|
| No guardrail | 67% | Made up answers to unknowable questions |
| Weak ("try to answer") | 33% | Inconsistent abstention |
| **Strong ("ONLY answer")** | **0%** | **Selected - clean abstention** |

### 4. Adversarial Testing (Day 8)
Built 17 test cases designed to break the system. **Found 1 vulnerability:** System was generating Python code when asked, violating scope. Fixed by adding explicit "do NOT write code" rule to system prompt.

## 🏃 Quick Start

### Prerequisites
- Python 3.11+
- Ollama installed ([ollama.com](https://ollama.com))

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull LLaMA model (first time only)
ollama pull llama3.2
```

### Run Pipeline
```bash
# Run RAG pipeline
python rag_pipeline.py

# Run all tests
pytest tests/ -v

# Run only fast mocked tests
pytest tests/test_rag_mocked.py -v

# View quality report
python quality_report.py
```

## 📂 Project Structure
```
genai_project/
├── rag_pipeline.py              # Main RAG implementation
├── quality_report.py            # Quality metrics summary
├── golden_dataset.py            # Benchmark Q&A pairs
├── requirements.txt             # Python dependencies
├── sample_docs/                 # Knowledge base documents
│   ├── refund.txt
│   ├── leave.txt
│   └── onboarding.txt
└── tests/
    ├── conftest.py              # pytest fixtures
    ├── test_rag_basic.py        # Integration tests
    ├── test_rag_mocked.py       # Unit tests (mocked)
    └── test_adversarial.py      # Adversarial tests
```

## 📈 Before vs After Optimization

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Hallucination Rate | 67% | 0% | Eliminated through strong guardrails |
| Retrieval Failures | 40% | 0% | Fixed via chunking optimization |
| Answer Quality | 60% | 100% | Systematic debugging + testing |
| Test Suite Speed | 85s | <1s | Added mocked unit tests |

## 🎓 Skills Demonstrated

- **RAG Architecture:** End-to-end pipeline design (ingestion → chunking → embedding → retrieval → generation)
- **Quality Evaluation:** RAGAS-style metrics (faithfulness, precision, recall, relevancy)
- **Systematic Debugging:** Root cause analysis for retrieval failures, context fragmentation, hallucinations
- **Test Automation:** pytest with fixtures, mocking, parametrization, adversarial scenarios
- **Prompt Engineering:** Experimental approach to guardrail optimization with measurable outcomes
- **LangChain:** LCEL chains, retrievers, prompt templates, document loaders, text splitters

## 🚧 Future Enhancements

- [ ] Migrate to Azure OpenAI + Azure AI Search (cloud deployment)
- [ ] Expand golden dataset from 5 to 50+ questions
- [ ] Add Flask API wrapper for production-like serving
- [ ] Implement CI/CD with GitHub Actions (automated testing on PR)
- [ ] Add Cosmos DB for conversation history (multi-turn chat)
- [ ] Deploy as Azure Function with Application Insights monitoring

## 📧 Contact

**[Your Name]**
- LinkedIn: www.linkedin.com/in/shubhangi-ajegaonkar-62aa76aa
- Email: shubhangi.ajegaonkar@gmail.com


---

*Built as part of GenAI testing specialization. Open to GenAI Tester / AI QA Engineer opportunities.*