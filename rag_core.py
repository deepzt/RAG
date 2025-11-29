import os
from typing import List, Tuple, Optional
import csv
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

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
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return embeddings_model


def get_reranker_model() -> CrossEncoder:
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder(reranker_model_name)
    return reranker_model


def create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    documents: List[Document] = []
    columns = df.columns.tolist()

    for idx, row in df.iterrows():
        parts = []
        for col in columns:
            val = row[col]
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        text = " | ".join(parts)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "row_index": int(idx),
                    "columns": ", ".join(columns),
                    "source_type": "table",
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

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
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

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunk_docs = splitter.split_documents(page_docs)

    for idx, doc in enumerate(chunk_docs):
        doc.metadata.setdefault("chunk_id", idx)

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)
    documents: List[Document] = []
    base_name = os.path.basename(path)
    for idx, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_type": "text",
                    "source_file": base_name,
                    "chunk_id": idx,
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


def build_context_and_answer(
    vector_store: FAISS,
    question: str,
    openai_model: Optional[ChatOpenAI],
    openai_model_name: Optional[str],
    ollama_model: Optional[ChatOllama],
    ollama_model_name: Optional[str],
) -> str:
    if not question.strip():
        return "Please enter a question."

    # First-stage retrieval: similarity search from FAISS
    try:
        k = 20
        results = vector_store.similarity_search_with_score(question, k=k)
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
    kept = rerank_results(question, results, top_k=3)

    # Derive a simple confidence signal from the reranker scores
    context_parts: List[str] = []
    best_score = None
    for i, (doc, score) in enumerate(kept, 1):
        if best_score is None or score > best_score:
            best_score = score
        content = doc.page_content
        if len(content) > 800:
            content = content[:800] + " ..."
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

    system_prompt = (
        "You are a careful assistant. Use ONLY the provided context to answer the user's "
        "question. If the answer is not clearly contained in the context, explicitly say "
        "'I do not know based on the provided context' and do not guess. "
        "Structure your response in exactly two sections:\n\n"
        "1) Under a heading 'Answer', provide a clear, concise explanation for the user. "
        "Use simple language, short paragraphs, and bullet points when helpful. Avoid "
        "jargon and do not repeat the question. Paraphrase the information from the "
        "context in your own words instead of copying it verbatim, except for short "
        "phrases or key terms that must be quoted.\n\n"
        "2) Under a heading 'Relevant context', list 1–3 bullet points showing which of the "
        "[Doc i - ...] snippets above are most relevant to the question and why. Do not "
        "introduce any information that is not in the provided context."
    )

    full_prompt = (
        f"{system_prompt}\n\n=== CONTEXT START ===\n{context_text}\n=== CONTEXT END ===\n\n"
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
        return answer
    except Exception as e:
        return (
            f"Model error: {e}\n\nHere is the retrieved context, which you can inspect manually:\n\n"
            + context_text
        )
