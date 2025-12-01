import os
import re
import tempfile
import math

import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

from rag_core import (
    get_embeddings_model,
    create_documents_from_dataframe,
    create_documents_from_pdf,
    create_documents_from_txt,
    get_openai_model,
    get_ollama_model,
    build_context_and_answer,
)


"""Gradio UI wrapper around the shared RAG core in rag_core.py."""


# Global cache for per-file vector stores
vector_store_cache = {}

# Load environment variables (for OPENAI_API_KEY, etc.)
load_dotenv()


def _history_to_markdown(history) -> str:
    """Render chat history (list of [user, assistant]) as Markdown."""
    if not history:
        return ""
    lines = []
    for pair in history:
        if not pair or len(pair) < 2:
            continue
        q, a = pair[0], pair[1]
        lines.append(f"**You:** {q}\n\n**Assistant:** {a}")
    return "\n\n---\n\n".join(lines)


def build_vector_store_for_file(file_path: str) -> tuple[FAISS, str]:
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
        raise ValueError("No content found in the uploaded file to index.")

    embeddings = get_embeddings_model()

    # Build FAISS index in batches for this file
    batch_size = 1024
    faiss_store = None
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        if faiss_store is None:
            faiss_store = FAISS.from_documents(batch_docs, embeddings)
        else:
            faiss_store.add_documents(batch_docs)

    vector_store = faiss_store

    build_info = (
        f"Built new FAISS index for {os.path.basename(file_path)} "
        f"from {source_desc} ({len(documents)} documents)."
    )
    return vector_store, build_info
openai_model, openai_model_name = get_openai_model()
ollama_model, ollama_model_name = get_ollama_model()


def on_file_upload(file):
    """Return a short status message when a file is uploaded."""
    if file is None:
        return "No file uploaded yet."
    return f"File uploaded: {os.path.basename(file.name)}. You can now ask questions."


def answer_question(file, question: str, history):
    """Gradio handler that streams status + answer and maintains chat history."""
    # Ensure history is always a list of [user, assistant] pairs (lists, not tuples)
    history = history or []

    if not question.strip():
        # Immediate feedback when no question is provided
        yield "Please enter a question.", _history_to_markdown(history), history
        return

    # Initial status while we start preparing the vector store
    yield "Preparing vector store... 0%", _history_to_markdown(history), history

    if file is None:
        yield "Please upload a file first.", _history_to_markdown(history), history
        return

    file_path = file.name
    try:
        stat = os.stat(file_path)
    except Exception as e:
        yield f"Error accessing uploaded file: {e}", _history_to_markdown(history), history
        return

    # Per-file cache key based on path and modification time
    cache_key = (file_path, stat.st_mtime)

    # Disk cache directory for FAISS indexes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_cache_root = os.path.join(base_dir, "faiss_cache")
    os.makedirs(faiss_cache_root, exist_ok=True)
    index_id = str(abs(hash(cache_key)))
    index_path = os.path.join(faiss_cache_root, index_id)

    vector_store = None
    status_msg = ""

    # 1) Try in-memory cache first
    if cache_key in vector_store_cache:
        vector_store = vector_store_cache[cache_key]
        status_msg = f"Using cached FAISS index for {os.path.basename(file_path)}. 100%"
        yield status_msg, _history_to_markdown(history), history
    else:
        # 2) Try loading from disk if index exists
        if os.path.exists(index_path):
            try:
                embeddings = get_embeddings_model()
                vector_store = FAISS.load_local(
                    index_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                vector_store_cache.clear()
                vector_store_cache[cache_key] = vector_store
                status_msg = (
                    f"Loaded FAISS index from disk for {os.path.basename(file_path)}. 100%"
                )
                yield status_msg, _history_to_markdown(history), history
            except Exception as e:
                # If loading fails, fall back to rebuilding
                yield f"Error loading existing index, rebuilding: {e}", _history_to_markdown(history), history
                vector_store = None

        # 3) Build a new index with progress if needed
        if vector_store is None:
            try:
                # Prepare documents for this file (including TXT support)
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
                    yield f"Unsupported file type: {ext}", _history_to_markdown(history), history
                    return

                if not documents:
                    yield "No content found in the uploaded file to index.", _history_to_markdown(history), history
                    return

                embeddings = get_embeddings_model()

                # Build FAISS index in batches with progress updates
                batch_size = 1024
                total_docs = len(documents)
                num_batches = math.ceil(total_docs / batch_size)
                faiss_store = None

                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, total_docs)
                    batch_docs = documents[start:end]

                    if faiss_store is None:
                        faiss_store = FAISS.from_documents(batch_docs, embeddings)
                    else:
                        faiss_store.add_documents(batch_docs)

                    progress = int(((batch_idx + 1) / num_batches) * 100)
                    yield (
                        f"Preparing vector store... {progress}% "
                        f"({start + 1}-{end} of {total_docs} documents)",
                        _history_to_markdown(history),
                        history,
                    )

                vector_store = faiss_store
                vector_store_cache.clear()
                vector_store_cache[cache_key] = vector_store

                # Persist index to disk for future sessions
                try:
                    vector_store.save_local(index_path)
                    status_msg = (
                        f"Built and saved FAISS index for {os.path.basename(file_path)} "
                        f"from {source_desc} ({total_docs} documents). 100%"
                    )
                except Exception as e:
                    status_msg = (
                        f"Built FAISS index for {os.path.basename(file_path)} "
                        f"from {source_desc} ({total_docs} documents), "
                        f"but failed to save index: {e}. 100%"
                    )

                yield status_msg, _history_to_markdown(history), history
            except Exception as e:
                yield f"Error building FAISS index: {e}", _history_to_markdown(history), history
                return

    if vector_store is None:
        yield "Error: vector store is not available.", _history_to_markdown(history), history
        return

    # Delegate retrieval, thresholding, logging, and answer generation
    # to the shared core so UI and CLI behave identically.
    try:
        yield status_msg + "\nRunning model...", _history_to_markdown(history), history
        answer = build_context_and_answer(
            vector_store,
            question,
            openai_model,
            openai_model_name,
            ollama_model,
            ollama_model_name,
            history,
        )
        # Update chat history with the new QA pair
        history = history or []
        history.append([question, answer])
        yield status_msg, _history_to_markdown(history), history
    except Exception as e:
        err_msg = f"Error while generating answer: {e}"
        history = history or []
        history.append([question, err_msg])
        yield status_msg, _history_to_markdown(history), history


def build_interface():
    description = (
        "Upload a CSV, XLS/XLSX, PDF, or TXT file. The app will build a semantic index over the "
        "file content and answer your questions using retrieved snippets plus a local Ollama/OpenAI model."
    )

    with gr.Blocks(title="File QA (CSV / Excel / PDF)") as demo:
        gr.Markdown("# File QA (CSV / Excel / PDF)")
        gr.Markdown(description)

        with gr.Row():
            file_input = gr.File(
                label="Upload file",
                file_types=[".csv", ".xls", ".xlsx", ".pdf", ".txt"],
            )

        upload_status = gr.Markdown(label="Upload status", value="No file uploaded yet.")
        chat_markdown = gr.Markdown(label="Conversation")
        question = gr.Textbox(label="Your question", lines=2)
        status = gr.Markdown(label="Status / Logs")

        history_state = gr.State([])

        file_input.change(fn=on_file_upload, inputs=file_input, outputs=upload_status)

        ask_btn = gr.Button("Ask")
        ask_btn.click(
            fn=answer_question,
            inputs=[file_input, question, history_state],
            outputs=[status, chat_markdown, history_state],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
