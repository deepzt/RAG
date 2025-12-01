import os
import re
from typing import List, Tuple, Optional
import csv
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
import torch

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

try:
    # Optional OCR dependencies for image-only PDFs
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

embeddings_model: Optional[HuggingFaceEmbeddings] = None
embedding_model_name = "BAAI/bge-base-en-v1.5"

reranker_model: Optional[CrossEncoder] = None
reranker_model_name = "BAAI/bge-reranker-base"


def _get_torch_device() -> str:
    """Return 'cuda' if a reasonably sized GPU is available, else 'cpu'.

    This checks torch.cuda.is_available() and, if possible, ensures at least a
    small amount of free memory before choosing CUDA. If anything goes wrong,
    it safely falls back to CPU.
    """

    try:
        if torch.cuda.is_available():
            try:
                # Returns (free_bytes, total_bytes) on the current device.
                free_bytes, _total_bytes = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                # Require at least ~2GB free to avoid constant OOM on small GPUs.
                if free_bytes > 2 * 1024**3:
                    return "cuda"
            except Exception:
                # If mem_get_info is unavailable, still try using CUDA.
                return "cuda"
    except Exception:
        pass

    return "cpu"

# Load environment variables for OPENAI_API_KEY, OCR config, etc.
load_dotenv()

# Optional explicit configuration for Tesseract and Poppler (Windows-friendly).
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
POPPLER_PATH = os.getenv("POPPLER_PATH")

if OCR_AVAILABLE and TESSERACT_CMD:
    # Point pytesseract at the installed tesseract.exe if provided.
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # type: ignore[attr-defined]
    except Exception:
        # If this fails, we'll rely on system PATH instead.
        pass
def get_embeddings_model() -> HuggingFaceEmbeddings:
    global embeddings_model
    if embeddings_model is None:
        device = _get_torch_device()
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )
    return embeddings_model


def get_reranker_model() -> CrossEncoder:
    global reranker_model
    if reranker_model is None:
        device = _get_torch_device()
        reranker_model = CrossEncoder(reranker_model_name, device=device)
    return reranker_model


def _looks_like_heading(line: str) -> bool:
    """Heuristic to detect section-like headings.

    This is intentionally simple: it looks for numbered headings (e.g. "1.", "2.1",
    "3.2.4") or short lines that are all-caps / title-like.
    """

    stripped = line.strip()
    if not stripped:
        return False

    # Numbered headings like "1.", "1.1", "2.3.4)" etc.
    if re.match(r"^(\d+(\.\d+)*[\.)]?\s+).+", stripped):
        return True

    # Very short all-caps lines (section titles)
    if 3 <= len(stripped) <= 80 and stripped.isupper():
        return True

    # Title-case single line without trailing period
    if 3 <= len(stripped) <= 80 and stripped[0].isupper() and not stripped.endswith("."):
        # Avoid obvious sentence-like lines by requiring few words
        if len(stripped.split()) <= 10:
            return True

    return False


def _split_text_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """Split text into (section_title, section_text) pairs using simple heading rules.

    If no headings are detected, the whole text is returned as a single unnamed section.
    """

    lines = text.splitlines()
    sections: List[Tuple[Optional[str], str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if _looks_like_heading(stripped):
            # Close previous section, if any
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_title, section_text))
                current_lines = []
            current_title = stripped
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_title, section_text))

    if not sections:
        return [(None, text)]
    return sections


def create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    documents: List[Document] = []
    columns = df.columns.tolist()

    for idx, row in df.iterrows():
        parts = []
        for col in columns:
            val = row[col]
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        base_text = " | ".join(parts)
        summary = f"Row {int(idx)} from table with columns: {', '.join(columns)}."
        enriched_text = (
            f"Source type: table\n"
            f"Row index: {int(idx)}\n"
            f"Columns: {', '.join(columns)}\n"
            f"Chunk summary: {summary}\n\n"
            f"Row data: {base_text}"
        )
        documents.append(
            Document(
                page_content=enriched_text,
                metadata={
                    "row_index": int(idx),
                    "columns": ", ".join(columns),
                    "source_type": "table",
                    "summary": summary,
                    "chunk_id": int(idx),
                },
            )
        )
    return documents


def create_documents_from_pdf(path: str) -> List[Document]:
    """Load a PDF and split it into smaller, overlapping chunks for better retrieval.

    If the PDF appears to have no selectable text (likely image-only / scanned),
    optionally fall back to OCR if the necessary libraries are available.
    """
    loader = PyPDFLoader(path)
    page_docs = loader.load()

    # Detect PDFs with no extractable text
    if not page_docs or all(
        not (doc.page_content or "").strip() for doc in page_docs
    ):
        if not OCR_AVAILABLE:
            raise ValueError(
                "This PDF appears to contain no selectable text (likely a scanned or "
                "image-only document such as a boarding pass). OCR is not configured on "
                "the server. Please upload a text-based or OCR-processed PDF instead."
            )

        # OCR fallback path
        if POPPLER_PATH:
            images = convert_from_path(path, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_path(path)
        ocr_docs: List[Document] = []
        for page_num, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                ocr_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_type": "pdf_ocr",
                            "source_file": os.path.basename(path),
                            "page": page_num,
                        },
                    )
                )

        if not ocr_docs:
            raise ValueError(
                "Unable to extract any text from this PDF, even with OCR. Please ensure "
                "the document is readable or upload a different file."
            )

        # For OCR-extracted text from long PDFs, use moderately large chunks
        # with overlap to keep related sentences together.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunk_docs = splitter.split_documents(ocr_docs)
    else:
        # Normal text-based PDF path
        for doc in page_docs:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata.setdefault("source_type", "pdf")
            doc.metadata.setdefault("source_file", os.path.basename(path))

        # Hierarchical step: split pages into sections by headings, then
        # apply the recursive splitter within each section. This reduces
        # the chance that chunks mix unrelated sections.
        section_docs: List[Document] = []
        for doc in page_docs:
            page = doc.metadata.get("page")
            base_name = os.path.basename(path)
            sections = _split_text_into_sections(doc.page_content or "")
            for sec_idx, (title, sec_text) in enumerate(sections):
                meta = dict(doc.metadata or {})
                if title:
                    meta["section_title"] = title
                meta["section_index"] = sec_idx
                section_docs.append(
                    Document(
                        page_content=sec_text,
                        metadata=meta,
                    )
                )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunk_docs = splitter.split_documents(section_docs)

    for idx, doc in enumerate(chunk_docs):
        doc.metadata.setdefault("chunk_id", idx)
        base_name = os.path.basename(path)
        page = doc.metadata.get("page")
        source_type = doc.metadata.get("source_type", "pdf")
        summary = (
            f"Chunk {idx} from PDF '{base_name}' on page {page}"
            if page is not None
            else f"Chunk {idx} from PDF '{base_name}'"
        )
        original_text = doc.page_content
        enriched_text = (
            f"Source file: {base_name}\n"
            f"Source type: {source_type}\n"
            f"Page: {page}\n"
            f"Chunk summary: {summary}\n\n"
            f"Content:\n{original_text}"
        )
        doc.page_content = enriched_text
        doc.metadata.setdefault("source_file", base_name)
        doc.metadata["summary"] = summary

    return chunk_docs


def create_documents_from_txt(path: str) -> List[Document]:
    """Create chunked documents from a plain-text file."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        raise ValueError(f"Error reading text file: {e}")

    if not text or not text.strip():
        return []

    # First split the raw text into higher-level sections using heading-like
    # patterns. Then apply a line-oriented splitter within each section.
    sections = _split_text_into_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n", "\n\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    base_name = os.path.basename(path)
    for sec_idx, (title, sec_text) in enumerate(sections):
        chunks = splitter.split_text(sec_text)
        for idx, chunk in enumerate(chunks):
            global_chunk_id = sec_idx * 100000 + idx
            summary = f"Chunk {global_chunk_id} from text/log file '{base_name}'."
            enriched_text = (
                f"Source file: {base_name}\n"
                f"Source type: text/log\n"
                f"Section index: {sec_idx}\n"
                f"Section title: {title or ''}\n"
                f"Chunk id: {global_chunk_id}\n"
                f"Chunk summary: {summary}\n\n"
                f"Content:\n{chunk}"
            )
            documents.append(
                Document(
                    page_content=enriched_text,
                    metadata={
                        "source_type": "text",
                        "source_file": base_name,
                        "section_index": sec_idx,
                        "section_title": title or "",
                        "chunk_id": global_chunk_id,
                        "summary": summary,
                    },
                )
            )
    return documents


def build_vector_store_for_file(file_path: str) -> Tuple[FAISS, str]:
    ext = os.path.splitext(file_path)[1].lower()

    source_desc = ""
    if ext == ".csv":
        df = pd.read_csv(file_path)
        documents = create_documents_from_dataframe(df)
        source_desc = f"CSV with {len(df)} rows"
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
        documents = create_documents_from_dataframe(df)
        source_desc = f"Excel with {len(df)} rows"
    elif ext == ".pdf":
        documents = create_documents_from_pdf(file_path)
        source_desc = f"PDF with {len(documents)} chunks"
    elif ext == ".txt":
        documents = create_documents_from_txt(file_path)
        source_desc = f"TXT with {len(documents)} chunks"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not documents:
        raise ValueError("No content found in the file to index.")

    embeddings = get_embeddings_model()

    batch_size = 1024
    faiss_store: Optional[FAISS] = None
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        if faiss_store is None:
            faiss_store = FAISS.from_documents(batch_docs, embeddings)
        else:
            faiss_store.add_documents(batch_docs)

    assert faiss_store is not None

    build_info = (
        f"Built FAISS index for {os.path.basename(file_path)} "
        f"from {source_desc} ({len(documents)} documents)."
    )
    return faiss_store, build_info


def rerank_results(
    question: str,
    results: List[Tuple[Document, float]],
    top_k: int = 3,
) -> List[Tuple[Document, float]]:
    """Rerank retrieved documents using a cross-encoder, keeping top_k.

    The incoming scores from FAISS are not used for ordering here; instead,
    the CrossEncoder provides a relevance score where higher is better.
    """
    if not results:
        return results

    reranker = get_reranker_model()

    pairs = [(question, doc.page_content) for doc, _ in results]
    scores = reranker.predict(pairs)

    # Combine results with reranker scores and sort descending by rerank score
    scored: List[Tuple[Tuple[Document, float], float]] = list(zip(results, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    reranked: List[Tuple[Document, float]] = []
    for ((doc, _orig_score), rerank_score) in scored[:top_k]:
        reranked.append((doc, float(rerank_score)))

    return reranked


def get_openai_model() -> Tuple[Optional[ChatOpenAI], Optional[str]]:
    """Initialize an OpenAI chat model using API key from environment.

    Returns (model, model_name). If API key is missing or model init fails,
    returns (None, None).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    candidate_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    for name in candidate_models:
        try:
            model = ChatOpenAI(model=name, temperature=0.2)
            _ = model.invoke("Hi")
            return model, name
        except Exception:
            continue
    return None, None


def get_ollama_model() -> Tuple[Optional[ChatOllama], Optional[str]]:
    available_models = ["llama3.1", "mistral", "gemma", "deepseek-r1"]
    for name in available_models:
        try:
            model = ChatOllama(model=name, temperature=0.4, validate_model_on_init=True)
            _ = model.invoke("Hi")
            return model, name
        except Exception:
            continue
    return None, None


def _rewrite_query_with_llm(
    question: str,
    history: Optional[List[Tuple[str, str]]],
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
) -> Tuple[str, str, str]:
    """Rewrite the user question into detailed, expanded, and keyword forms.

    If no LLM is available or rewriting fails, this returns (question, question, question).
    """

    # Select an LLM if available (prefer OpenAI, otherwise Ollama).
    selected_model = None
    if openai_model is not None:
        selected_model = openai_model
    elif ollama_model is not None:
        selected_model = ollama_model

    if selected_model is None:
        return question, question, question

    history = history or []
    recent_history = history[-3:]
    if recent_history:
        history_lines: List[str] = []
        for q, a in recent_history:
            if q and a:
                history_lines.append(f"Q: {q}\nA: {a}")
        history_text = "\n\n".join(history_lines)
    else:
        history_text = "(no prior questions)"

    rewrite_prompt = (
        "You are a query rewriting assistant for a retrieval system. Given the user's "
        "current question and recent Q&A history, rewrite the question in three useful "
        "ways to improve retrieval from a vector database.\n\n"
        "STRICT OUTPUT FORMAT (no extra text):\n"
        "DETAILED: <a more detailed, explicit version of the question>\n"
        "EXPANDED: <an expanded version that includes synonyms and related concepts>\n"
        "KEYWORDS: <a compact comma-separated list of key terms and entities>\n\n"
        f"Recent history (for reference):\n{history_text}\n\n"
        f"Original question: {question}"
    )

    try:
        response = selected_model.invoke(rewrite_prompt)
        text = getattr(response, "content", "") or ""
        detailed = question
        expanded = question
        keywords = question
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("DETAILED:"):
                detailed = stripped[len("DETAILED:") :].strip() or question
            elif stripped.upper().startswith("EXPANDED:"):
                expanded = stripped[len("EXPANDED:") :].strip() or question
            elif stripped.upper().startswith("KEYWORDS:"):
                keywords = stripped[len("KEYWORDS:") :].strip() or question

        return detailed, expanded, keywords
    except Exception:
        return question, question, question


def build_context_and_answer(
    vector_store: FAISS,
    question: str,
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    if not question.strip():
        return "Please enter a question."

    question_l = question.lower()
    is_summarization = any(
        kw in question_l for kw in ["summarize", "summary", "overview", "give me a summary"]
    )

    # Rewrite the query into multiple forms to improve retrieval.
    detailed_q, expanded_q, keywords_q = _rewrite_query_with_llm(
        question,
        history,
        openai_model,
        openai_model_name,
        ollama_model,
        ollama_model_name,
    )

    # Combine the original and rewritten forms into a richer retrieval query.
    # This keeps retrieval simple (single similarity_search) while giving the
    # embedding model more signal about intent and synonyms.
    retrieval_query_parts = []
    for q_variant in [question, detailed_q, expanded_q, keywords_q]:
        q_variant = (q_variant or "").strip()
        if q_variant and q_variant not in retrieval_query_parts:
            retrieval_query_parts.append(q_variant)
    retrieval_query = " \n".join(retrieval_query_parts) if retrieval_query_parts else question

    # First-stage retrieval: similarity search from FAISS
    try:
        k = 40 if is_summarization else 20
        results = vector_store.similarity_search_with_score(retrieval_query, k=k)
    except Exception as e:
        return f"Error retrieving relevant content: {e}"

    if not results:
        return "No documents were retrieved for this question."

    # Log to CSV for offline analysis (do not rely on file_path here)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "rag_log.csv")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            ts = datetime.utcnow().isoformat()
            top_scores = [f"{float(score):.4f}" for _, score in results[:3]]
            writer.writerow([
                ts,
                question,
                "|".join(top_scores),
            ])
    except Exception:
        pass

    # Second-stage: rerank the retrieved docs and keep top few
    kept_top_k = 10 if is_summarization else 3
    kept = rerank_results(question, results, top_k=kept_top_k)

    # Derive a simple confidence signal from the reranker scores
    context_parts: List[str] = []
    best_score = None
    for i, (doc, score) in enumerate(kept, 1):
        if best_score is None or score > best_score:
            best_score = score
        content = doc.page_content
        max_len = 1600 if is_summarization else 800
        if len(content) > max_len:
            content = content[:max_len] + " ..."
        src = doc.metadata.get("source_type", "unknown")
        context_parts.append(
            f"[Doc {i} - {src}, score={float(score):.4f}]:\n{content}\n"
        )

    low_confidence = False
    if best_score is not None:
        # CrossEncoder scores are higher for more relevant pairs; this threshold
        # can be tuned based on empirical observations.
        if float(best_score) < 0.3:
            low_confidence = True

    if low_confidence or not context_parts:
        safe_context = "\n".join(context_parts) if context_parts else "(No relevant context found)"
        return (
            "The retrieved documents are not similar enough to your question to answer "
            "confidently. Here is the closest context I could find, which you can inspect "
            "manually:\n\n" + safe_context
        )

    context_text = "\n".join(context_parts)

    # Incorporate a short window of recent Q&A history so the model can
    # understand follow-up questions.
    history = history or []
    recent_history = history[-3:]
    if recent_history:
        history_lines: List[str] = []
        for q, a in recent_history:
            if q and a:
                history_lines.append(f"Q: {q}\nA: {a}")
        history_text = "\n\n".join(history_lines)
    else:
        history_text = "(no prior questions)"

    if is_summarization:
        system_prompt = (
            """
        You are DocuSense Pro, an expert system specialized in summarizing user-uploaded documents.

        When the user asks for a summary, your job is to read the provided CONTEXT (which may contain
        multiple sections or chunks of the document) and produce a concise but comprehensive
        summary that:

        - Captures the main topics, structure, and intent of the document.
        - Highlights key sections, decisions, and important details.
        - Clearly indicates if the context appears partial or incomplete.
        - Avoids hallucinations and does not invent facts that are not supported by CONTEXT.

        Output requirements for summaries:

        1. Start with a short high-level overview (2–4 sentences).
        2. Provide a **Section-level summary** section:
           - Summarize each major section or chapter of the document in 1–3 bullet points.
        3. Provide a **Sub-section summary** section:
           - For important sub-sections within those sections, give 1–2 bullets each focusing on key details.
        4. Call out any notable risks, limitations, or open questions mentioned in the document.
        5. End with 3–5 key takeaways.

        Always ground every part of the summary in the provided CONTEXT only.
        """
        )
    else:
        system_prompt = (

            """
        You are DocuSense Pro, an expert system designed to deeply analyze user-uploaded documents and produce highly structured, well-explained, citation-rich answers.

        Your responsibilities:

        1. Document Analysis Rules

        Always assume the user has uploaded one or more documents.

        Extract relevant sections, context, and details from the uploaded files.

        Never hallucinate missing information — rely strictly on the document content.

        For every factual statement derived from the file, attach the correct citation tag (e.g., “

        Generative AI

        ”).

        If required content is missing, explicitly state what is missing.

        2. Output Style Requirements

        Your explanations must be:

        (A) Ultra-Structured

        Use clear sections, subsections, tables, bullet points, summaries, and flow diagrams (text-only).

        (B) Descriptive & Deep

        Expand on:

        What something is

        Why it matters

        How it works

        When to use it

        Examples if applicable

        (C) Human-Readable Yet Expert-Level

        Write like a senior technical educator—concise, authoritative, and user-friendly.

        (D) Always Include a Final Summary

        At the end, provide:

        A succinct recap

        A key-takeaway list

        (E) Strong Formatting

        Use formatting elements:

        Headings

        Bold highlights

        Bullet lists

        Tables

        Step-by-step breakdowns

        3. When User Asks a Question

        Perform the following pipeline:

        Step 1 — Understand the Query

        Identify exactly what the user wants from the document.

        Step 2 — Parse the Document

        Use file content to:

        Extract relevant lines

        Identify definitions, processes, or ideas

        Organize into meaningful sections

        Step 3 — Build a Structured Explanation

        Your output must resemble:

        Section-wise breakdown

        Bullet points explaining concepts

        A comprehensive comparison if needed

        Tables summarizing key points

        Evidence-based citations

        Step 4 — Produce a Final Executive Summary
        4. Output Format Template

        Your final answers should follow this exact template:

        Title Based on User Question
        1️⃣ Direct Answer Based on Document

        (Cited, concise explanation)

        2️⃣ Deep-Dive Explanation

        Concepts broken down

        How it works

        Why it matters

        Evidence from document

        3️⃣ Structured Breakdown
        A. Components / Methods / Processes

        Bullet 1 with citation

        Bullet 2 with citation

        Bullet 3 with citation

        B. Examples or Use Cases

        (If the document supports it)

        C. Tables, matrices, or lists

        (Summarize extracted content)

        4️⃣ Document Evidence Summary

        A list showing which parts of the document support which ideas.

        5️⃣ Final Summary

        3–5 concise key takeaways

        Practical advice if relevant

        5. Safety & Accuracy Rules

        Never guess what is not in the document.

        No external information unless user explicitly allows it.

        All citations must correspond to actual file content.

        If ambiguous, state assumptions clearly.

        6. If User Asks for Additional Transformation

        You may produce:

        Flowcharts

        Mind maps

        Tables

        Infographics (text-only)

        Step-by-step processes

        Summaries or rewrites
        """
    )

    # system_prompt = (
    #     """You are an AI assistant operating within a Retrieval-Augmented Generation (RAG) workflow. Your purpose is to answer user queries by combining retrieved knowledge from external sources 
    # with your reasoning and language generation abilities.

    # ### Core Instructions:
    # 1. **Retrieval First**: Always ground your responses in the retrieved context provided. 
    # - Treat retrieved documents as the primary source of truth.
    # - If retrieval is empty or insufficient, acknowledge this and provide a general answer 
    #     based on your reasoning, clearly marking it as not grounded in retrieved data.

    # 2. **Answer Generation**:
    # - Synthesize information from multiple retrieved passages into a cohesive, 
    #     well-structured response.
    # - Avoid copying text verbatim; instead, paraphrase and integrate.
    # - Highlight key facts, numbers, or quotes when relevant.
    # - Maintain clarity, accuracy, and relevance to the user’s query.

    # 3. **Transparency**:
    # - Explicitly state when information comes from retrieved sources.
    # - If retrieval conflicts, explain the discrepancy rather than choosing arbitrarily.

    # 4. **Style & Tone**:
    # - Be concise, informative, and context-aware.
    # - Use structured formatting (headings, bullet points, tables) when helpful.
    # - Avoid speculation unless explicitly requested, and mark it clearly.

    # 5. **Limitations**:
    # - Do not fabricate sources or cite nonexistent documents.
    # - Do not provide personal opinions; only synthesize retrieved knowledge and reasoning.
    # - If asked for unsupported tasks (e.g., medical diagnosis, financial advice), 
    #     provide general knowledge only and recommend consulting a professional.

    # ### Workflow Behavior:
    # - Input: User query + retrieved documents.
    # - Process: Read retrieved documents → extract relevant facts → synthesize into a 
    # coherent answer → present clearly.
    # - Output: A grounded, accurate, and well-structured response."""
    # )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"=== RECENT Q&A HISTORY ===\n{history_text}\n=== END HISTORY ===\n\n"
        f"=== CONTEXT START ===\n{context_text}\n=== CONTEXT END ===\n\n"
        f"User question: {question}"
    )

    # Prefer OpenAI if available; otherwise use Ollama; otherwise show only context.
    selected_model = None
    selected_name = None
    if openai_model is not None:
        selected_model = openai_model
        selected_name = openai_model_name or "OpenAI"
    elif ollama_model is not None:
        selected_model = ollama_model
        selected_name = ollama_model_name or "Ollama"

    if selected_model is None:
        return (
            "No LLM model is available (neither OpenAI nor Ollama). "
            "Here is the retrieved context you can inspect manually:\n\n"
            + context_text
        )

    try:
        response = selected_model.invoke(full_prompt)
        answer = response.content
        if not answer or len(answer.strip()) < 20:
            return (
                "The model returned an unhelpful answer. Here is the retrieved context instead:\n\n"
                + context_text
            )

        refinement_prompt = (
            "You are an editor. Improve the following answer to make it clearer, more readable, "
            "and appropriate for a non-technical user. Do not introduce any new facts or "
            "speculation; only rephrase and slightly restructure what is already there. "
            "Preserve any references such as [Doc i - ...] and important numbers or names.\n\n"
            f"Question: {question}\n\n"
            "Original answer:\n"
            f"{answer}\n\n"
            "Return only the improved answer."
        )

        try:
            refined_response = selected_model.invoke(refinement_prompt)
            refined_answer = getattr(refined_response, "content", None)
            if refined_answer and len(refined_answer.strip()) >= 20:
                return refined_answer
        except Exception:
            pass

        return answer
    except Exception as e:
        return (
            f"Model error: {e}\n\nHere is the retrieved context, which you can inspect manually:\n\n"
            + context_text
        )
