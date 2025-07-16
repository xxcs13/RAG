# RAG System

A Retrieval-Augmented Generation (RAG) system that processes multiple document formats with retrieval and answering capabilities.

## Features

### Multi-Format Document Processing
- **PDF Documents**: Chunking with 300-token segments and 50-token overlap
- **PowerPoint Presentations**: Slide-by-slide processing with table extraction
- **Excel Spreadsheets**: Row-integrity preserving chunking with header context

### Advanced Retrieval Pipeline
- **Index Routing**: Smart file-type selection based on query content
- **Vector Search**: Semantic similarity search using OpenAI embeddings
- **Parent Aggregation**: Deduplication of chunks from the same source unit
- **LLM Reranking**: GPT-4o-mini powered relevance scoring with score fusion
- **Context Assembly**: Structured context preparation with source identification

### Intelligent Answering
- **Structured Output**: JSON-formatted responses with step-by-step analysis
- **Chain-of-Thought**: Detailed reasoning process in answers
- **Complex Query Handling**: Automatic query decomposition for comparison questions
- **Source Citation**: Automatic citation of relevant document sections

## Architecture

### Core Components
1. **Document Ingestion**: Multi-format parsing with format-specific chunking strategies
2. **Vector Storage**: ChromaDB-based persistent vector database
3. **Retrieval Engine**: Multi-stage retrieval with routing and reranking
4. **Answer Generation**: GPT-4o powered structured response generation
5. **Workflow Management**: LangGraph-based state management

### Key Design Principles
- **Modular Architecture**: Separate nodes for ingestion, embedding, retrieval, and generation
- **Smart Chunking**: Format-aware chunking that preserves document structure
- **Hybrid Scoring**: Combination of vector similarity and LLM relevance scores
- **Robust Error Handling**: Graceful fallbacks for parsing and API failures

## Technical Implementation

### Document Processing Pipeline
The system implements a sophisticated document processing pipeline that handles heterogeneous document formats through specialized parsers. Each parser employs format-specific strategies to maintain semantic coherence while optimizing for retrieval performance.

### Retrieval Strategy
The retrieval mechanism incorporates a multi-stage approach combining vector similarity search with neural reranking. The system utilizes index routing to reduce computational overhead and improve precision by targeting relevant document types based on query analysis.

### Answer Generation Framework
The answer generation component leverages large language models with structured prompting techniques. Complex queries are automatically decomposed into sub-queries, processed independently, and synthesized into comprehensive responses with full source attribution.

## Prerequisites

- Python 3.10.18
- OpenAI API key
- Conda environment manager

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n rag python=3.10
conda activate rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 4. Prepare Documents
Ensure the following files are in the project directory:
- `tsmc_2024_yearly_report.pdf`
- `無備忘錄版_digitimes_2025年全球AI伺服器出貨將達181萬台　高階機種採購不再集中於四大CSP.pptx`
- `excel_FS-Consolidated_1Q25.xls`

## Usage

### Basic Usage
```bash
python rag.py
```

The system will:
1. Parse and embed all documents (one-time setup)
2. Present an interactive query interface
3. Process questions with enhanced retrieval and generation

### Query Examples
- **Simple Questions**: "What is TSMC's revenue?"
- **Complex Comparisons**: "Compare AI server shipments between regions"
- **Multi-source Queries**: "Analyze financial performance and market trends"

## System Output

The system provides structured responses including:
- **Selected Indexes**: Which document types were searched
- **Confidence Level**: High/Medium/Low based on available information
- **Step-by-Step Analysis**: Detailed reasoning process
- **Relevant Sources**: Specific document sections used
- **Final Answer**: Comprehensive response to the query

## Configuration

### Key Parameters
- **Chunk Size**: 300 tokens (configurable)
- **Overlap Size**: 50 tokens (configurable)
- **Retrieval Count**: Top 30 candidates per index
- **Final Selection**: Top 10 reranked parents
- **Score Fusion**: 30% vector + 70% LLM weighting

### Model Configuration
- **Embeddings**: OpenAI text-embedding-3-large
- **Reranking**: GPT-4o-mini
- **Generation**: GPT-4o
- **Vector Store**: ChromaDB with persistence

## File Structure

```
rag/
├── rag.py                         # Main application
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
├── README.md                      # This file
├── chroma_db_new/                 # Vector database (auto-created)
├── rag_qa_log.csv                 # Query logs (auto-created)
└── [document files]               # Input documents
```

## Key Functions

### Document Processing
- `parse_pdf()`: PDF text extraction with intelligent chunking
- `parse_pptx()`: PowerPoint content extraction with table support
- `parse_xls()`: Excel processing with row-integrity preservation

### Retrieval Pipeline
- `route_indexes()`: Query-based document type selection
- `llm_rerank_batch()`: GPT-4o-mini powered relevance scoring
- `aggregate_parents()`: Duplicate source elimination

### Response Generation
- `handle_simple_query()`: Direct question answering
- `handle_complex_query()`: Multi-part query decomposition
- `synthesize_sub_results()`: Complex answer synthesis

## Performance Features

- **Persistent Storage**: Vector embeddings cached for reuse
- **Batch Processing**: Efficient LLM API usage
- **Smart Routing**: Reduced search space through index selection
- **Comprehensive Logging**: Detailed metrics and debugging information


## Logging

The system automatically logs:
- Question and answer pairs
- Retrieval metrics and source information
- Confidence levels and analysis steps
- Performance and debugging data

## Technical Specifications

### Chunking Strategy
The system employs a token-based chunking approach with configurable parameters to balance between semantic coherence and retrieval granularity. Document-specific strategies ensure optimal information preservation across different formats.

### Vector Database
ChromaDB provides persistent vector storage with efficient similarity search capabilities. The system maintains document metadata and supports incremental updates for new document additions.

### Query Processing
Query processing incorporates natural language understanding to route queries to appropriate document subsets, reducing computational overhead and improving response accuracy.

### Response Quality Assurance
All generated responses undergo validation through structured output schemas and confidence scoring mechanisms to ensure reliability and accuracy.

---

**Note**: This system requires an active OpenAI API key and internet connection for optimal performance.
