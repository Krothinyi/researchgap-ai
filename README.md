## Live Demo

**[Launch ResearchGap AI — Public Demo](https://researchgap-ai-2layxqv7aqvwgqkmyzpggb.streamlit.app/)**

Explore research papers, compare methodologies, identify potential research gaps, and ask evidence-backed questions using RAG and Gemini.

The public demo includes two preloaded research papers.

**[View Demo Source Code](https://github.com/Krothinyi/researchgap-ai/tree/feature/portfolio-demo)**

### Full Interactive Version

The full version of ResearchGap AI supports PDF uploads, automatic paper downloading and indexing, research comparison, and document management.

**Access to the deployed full application is restricted to authorized users.** Its source code is maintained in the `main` branch.

### AI-Powered Research Paper Discovery, Analysis, and Comparison

ResearchGap AI is a research assistant built with Python,
Streamlit, Qdrant, and Google's Gemini API.

It helps researchers discover academic papers, explore their
contents, compare research methodologies, and investigate
potential research gaps using evidence retrieved from
scientific literature.

The goal is to make literature exploration more efficient,
accessible, and evidence-driven.

## Features

### 1. Research Paper Discovery
- Search academic literature using natural-language queries.
- Discover research papers through the OpenAlex API.
- View publication metadata and available PDF links.
- Select relevant papers for further analysis.

### 2. PDF Download and Indexing
- Download available research PDFs.
- Manually upload research papers.
- Extract text from PDF documents.
- Split documents into searchable passages.
- Generate vector embeddings.
- Store and retrieve embeddings using Qdrant.
- Manage indexed papers.

### 3. AI-Powered Paper Comparison
- Compare selected research papers.
- Analyze methodologies, findings, and limitations.
- Identify similarities and differences.
- Generate evidence-backed comparative summaries.
- Explore potential research gaps.

### 4. Research Question Answering
- Ask natural-language questions about research papers.
- Search one selected paper or all indexed papers.
- Retrieve relevant passages using semantic search.
- Generate answers using Gemini and retrieved evidence.
- Display supporting research sources.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Research Discovery | OpenAlex API |
| Language Model | Google Gemini |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | Qdrant |
| PDF Processing | PyMuPDF |
| Data Validation | Pydantic |

## How ResearchGap AI Works

1. The user enters a research topic or question.
2. The application searches for related papers through OpenAlex.
3. The user selects papers to download or upload manually.
4. The application extracts and chunks the document text.
5. Sentence Transformers generates vector embeddings.
6. Qdrant stores and retrieves relevant passages.
7. Gemini uses retrieved evidence to generate answers,
   comparisons, and potential research directions.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/researchgap-ai.git
cd researchgap-ai
```

Replace `YOUR_USERNAME` with the repository owner's
GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Python 3.11 or newer is required by the pinned PyMuPDF
dependency.

### 4. Configure Gemini

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Obtain a Gemini API key from Google AI Studio.

Never commit your actual API key to GitHub.

### 5. Run ResearchGap AI

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

http://localhost:8501

The application uses a local Qdrant database. It initializes
a collection if one does not already exist.

## Example Use Cases

**Paper discovery**

"Find research papers on hallucination detection in
large language models."

**Research comparison**

"Compare the methodologies and limitations of two
retrieval-augmented generation approaches."

**Research question answering**

"What are the main limitations discussed in this paper?"

**Research gap exploration**

"What potential research directions emerge from
these papers?"

## Limitations

- Paper availability depends on accessible research sources.
- Some academic papers may not have downloadable PDFs.
- Scanned PDFs may not support reliable text extraction.
- Generated explanations can contain errors or omissions.
- Research gaps are suggestions, not verified claims of novelty.
- API usage may be subject to provider limits and charges.
- The application currently uses local Qdrant storage.

Users should verify important claims against the original
research papers.

## Privacy and Data

Downloaded PDFs, extracted text, embeddings, and the local
Qdrant database are excluded from the Git repository.

Research queries and relevant retrieved evidence may be sent
to external services, including OpenAlex and Google's Gemini
API, as part of the application's operation.

Users should avoid uploading confidential research material
without considering the relevant provider policies.

## Project Status

ResearchGap AI is an actively developed MVP focused on
research discovery, semantic question answering, and
evidence-supported literature comparison.

Future work may include improved evidence verification,
expanded document processing, and deployment support.
