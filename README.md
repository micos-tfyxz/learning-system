# Learning System: Document Analysis Assistant

## Overview

The **Learning System** is a powerful document analysis assistant designed to process PDF documents and extract meaningful insights. Using advanced AI technologies such as GPT-4 and vector embeddings, the system transforms documents into structured, searchable, and interactive content. The system also provides a user-friendly interface for querying and generating reports.

---

## Core Workflow

### 1. PDF Ingestion
- **Input**: Users upload a PDF document through the system.
- **Processing**: Extracts raw text, tables, and images along with their positional metadata using libraries like `pdfplumber` and `PyMuPDF`.

---

### 2. Content Structuring
- **Text**: 
  - Hierarchical parsing with heading detection based on font size, position, and styling (bold/italic).
- **Tables**:
  - Matches captions to table bounding boxes.
  - Extracts positional metadata for tables.
- **Images**:
  - Extracts coordinates and associates images with corresponding captions.

---

### 3. Semantic Processing
For all content types (text, tables, and images):
- **AI-Generated Descriptions**:
  - Uses GPT-4 to generate concise and meaningful descriptions for text, tables, and images.
- **Vector Embeddings**:
  - Converts content into vector embeddings using `text-embedding-3-small` for semantic search.
- **Unified Format**:
  - Standardizes all content types using the `ContentBase` model.

---

### 4. Vector Database Storage
- **MongoDB** stores:
  - Original content (raw text, images, tables).
  - AI-generated descriptions.
  - Vector embeddings for semantic search.
  - Metadata such as page numbers and positional information.

---

### 5. Query Interface
- **Streamlit Frontend**:
  - Accepts natural language queries from users.
- **Semantic Search**:
  - Finds relevant content based on vector similarity.
- **AI-Powered Output**:
  - GPT-4 generates:
    - Structured reports in Markdown format.
    - Follow-up questions to deepen understanding.
    - Contextual summaries of relevant content.

---

### 6. Output Generation
- **Interactive Reports**:
  - Include source references (e.g., page numbers).
  - Multi-modal content display (text, images, and tables).
  - Allows users to explore query results interactively.
- **Query History Tracking**:
  - Tracks and maintains a history of user-generated reports for easy access.

---

## Key Features
- **Multi-Modal Content Processing**: Handles text, images, and tables seamlessly.
- **AI-Powered Descriptions**: Summarizes and contextualizes content using GPT-4.
- **Semantic Search**: Allows users to find relevant content using natural language.
- **Interactive Reports**: Provides a rich and interactive experience with Markdown-based reports.
- **Scalable Storage**: Stores content and metadata in MongoDB for efficient querying.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/micos-tfyxz/learning-system.git
   cd learning-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**:
   - Install and configure MongoDB.
   - Ensure it is running on `localhost:27017`.

4. **Set up OpenAI API**:
   - Create an OpenAI account and generate an API key.
   - Set the `OPENAI_API_KEY` environment variable:
     ```bash
     export OPENAI_API_KEY=your_openai_api_key
     ```

5. **Run the application**:
   ```bash
   streamlit run frontend\ demo.py
   ```

---

## example
1. Analyze the PDF using main.py and check if the results are stored in MongoDB,
![image](https://github.com/user-attachments/assets/8d65ffc8-2a7a-4272-b575-22ae9ef2a967)

2.open the frontend demo.py to ask questions related to the PDF, and it will generate answers and provide some relevant questions for selection

---

## Technologies Used
- **Python Libraries**:
  - `pdfplumber`, `PyMuPDF`: PDF processing.
  - `openai`: GPT-4 and embeddings.
  - `numpy`: Vector operations.
  - `streamlit`: Frontend interface.
  - `pymongo`: MongoDB integration.
- **AI Models**:
  - `GPT-4` for descriptions, summaries, and follow-up questions.
  - `text-embedding-3-small` for vector embeddings.

---

## Project Structure
```
learning-system/
├── datamodels.py          # Standardized data models
├── query_service.py       # Query processing and report generation
├── picture.py             # Image extraction and processing
├── mongodb_store.py       # MongoDB integration
├── main.py                # Main workflow integration
├── frontend demo.py       # Streamlit frontend
├── vectorizer.py          # Vectorization service
├── table.py               # Table extraction and processing
├── text.py                # Text extraction and summarization
```

---
