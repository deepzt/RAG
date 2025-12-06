import os
import re
import tempfile
import math
import threading
import queue
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr
import pyttsx3

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


# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech

# Global variables for voice control
engine = None
stop_speaking = False

# Function to stop voice output
def stop_voice():
    global engine, stop_speaking
    stop_speaking = True
    if engine:
        try:
            engine.stop()
            engine = None  # Reset the engine
        except:
            pass
    return ""

# Function to speak text using pyttsx3
def _speak_text(text):
    global engine, stop_speaking
    stop_speaking = False  # Reset the stop flag
    try:
        if engine is None:
            engine = pyttsx3.init()
        
        # Split text into sentences for better interruption handling
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if stop_speaking:
                break
            if engine is None:  # In case engine was stopped
                engine = pyttsx3.init()
            engine.say(sentence)
            engine.runAndWait()
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
    finally:
        if stop_speaking:  # If stopped, reset the engine
            try:
                if engine:
                    engine.stop()
            except:
                pass
            engine = None

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
    """Handle file upload and return file path, vector store, and status."""
    if file is None:
        return None, None, "No file uploaded yet."
    try:
        # Return the file path as a string, not a file object
        file_path = file.name
        vs, info = build_vector_store_for_file(file_path)
        return file_path, vs, f"Successfully processed {os.path.basename(file_path)}. You can now ask questions."
    except Exception as e:
        return None, None, f"Error processing file: {str(e)}"


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


def record_audio(duration=5, fs=44100):
    """Record audio from the microphone using sounddevice."""
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save the recording to a temporary file
    temp_file = tempfile.mktemp(prefix='voice_input_', suffix='.wav')
    write(temp_file, fs, recording)  # Save as WAV file
    
    return temp_file

def listen_microphone():
    """Listen to microphone and return the recognized text using sounddevice for recording."""
    try:
        # Record audio using sounddevice
        audio_file = record_audio(duration=5)
        
        # Use SpeechRecognition to transcribe the audio file
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Recognized: {text}")
            return text
            
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with the speech recognition service; {e}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def build_interface():
    description = (
        "Upload a CSV, XLS/XLSX, PDF, or TXT file. The app will build a semantic index over the "
        "file content and answer your questions using retrieved snippets plus a local Ollama/OpenAI model."
    )

    with gr.Blocks(title="RAG Chat") as demo:
        gr.Markdown("# RAG Chat Application with Voice")
        
        with gr.Row():
            with gr.Column(scale=3):
                file_output = gr.File(label="Upload a document")
                # File upload button
                file_upload = gr.UploadButton(
                    "📁 Upload Document",
                    file_types=[".txt", ".pdf", ".csv", ".xls", ".xlsx"],
                    file_count="single",
                )
                
                # File upload handler will be defined after the components

                model_choice = gr.Radio(
                    ["OpenAI (if API key available)", "Local (Ollama)"],
                    label="Choose AI Model",
                    value="Local (Ollama)",
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Creativity (temperature)",
                )
                
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Number of document chunks to retrieve",
                )
                
                chunk_size = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Chunk size (tokens)",
                )
                
                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=200,
                    step=50,
                    label="Chunk overlap (tokens)",
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    voice_input_btn = gr.Button("🎤 Voice Input")
                
                # Voice output toggle
                voice_output = gr.Checkbox(
                    label="Enable Voice Output",
                    value=True,
                    info="Read responses aloud"
                )
                
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=500,
                    avatar_images=(
                        "https://i.ibb.co/8M7fB6Y/user.png",
                        "https://i.ibb.co/0XyJtJz/assistant.png",
                    ),
                )
                
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Type your question or click the microphone to speak...",
                    container=False,
                    scale=7,
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    
                    # Connect stop button
                    stop_btn.click(
                        fn=stop_voice,
                        inputs=None,
                        outputs=[],
                        queue=False  # Don't queue stop commands
                    )
                
                gr.Examples(
                    examples=[
                        "Summarize the key points from this document.",
                        "What is the main topic of this document?",
                        "Find relevant information about...",
                    ],
                    inputs=msg,
                    label="Example questions",
                )
        
        # State to store the current file and vector store
        current_file = gr.State()
        vector_store = gr.State()
        
        # Function to handle voice input
        def on_voice_input():
            try:
                text = listen_microphone()
                return text
            except Exception as e:
                print(f"Error in voice input: {e}")
                return "Error processing voice input"
        
        # Event handler for voice input button
        voice_input_btn.click(
            fn=on_voice_input,
            inputs=[],
            outputs=[msg]
            )

        upload_status = gr.Markdown(label="Upload status", value="No file uploaded yet.")
        chat_markdown = gr.Markdown(label="Conversation")
        question = gr.Textbox(label="Your question", lines=2)
        status = gr.Markdown(label="Status / Logs")

        history_state = gr.State([])

        # Single file upload handler
        def handle_file_upload(file):
            if file is None:
                return None, None, "No file uploaded yet."
            
            try:
                # Process the file and build the vector store
                vs, info = build_vector_store_for_file(file.name)
                
                # Return the file object, vector store, and status message
                return (
                    file,  # Return the file object for display
                    vs,    # The vector store for querying
                    f"Successfully processed {os.path.basename(file.name)}. You can now ask questions."
                )
            except Exception as e:
                return None, None, f"Error processing file: {str(e)}"
        
        # Connect the file upload handler
        file_upload.upload(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[file_output, vector_store, upload_status]
        )

        # Handle question submission
        def on_ask_click(question, history, vs, model_choice_val):
            if not question.strip():
                return "Please enter a question.", history or [], None
            if vs is None:
                return "Please upload a document first.", history or [], None
                
            try:
                # Get the appropriate model based on user choice
                if model_choice_val == "OpenAI (if API key available)":
                    if openai_model is None:
                        return "OpenAI API key not found. Please check your .env file or switch to Local (Ollama).", history, None
                    model = openai_model
                    model_name = openai_model_name
                else:
                    if ollama_model is None:
                        return "Ollama model not available. Make sure Ollama is running and the model is downloaded.", history, None
                    model = ollama_model
                    model_name = ollama_model_name
                
                # Generate response
                response = build_context_and_answer(
                    vector_store=vs,
                    question=question,
                    openai_model=openai_model if model_choice_val == "OpenAI (if API key available)" else None,
                    openai_model_name=openai_model_name if model_choice_val == "OpenAI (if API key available)" else None,
                    ollama_model=ollama_model if model_choice_val == "Local (Ollama)" else None,
                    ollama_model_name=ollama_model_name if model_choice_val == "Local (Ollama)" else None,
                    history=history,
                )
                
                # Update chat history with properly formatted messages
                history = history or []
                # Add user message
                history.append({"role": "user", "content": question})
                # Add assistant message
                history.append({"role": "assistant", "content": response})
                
                # Speak the response if voice output is enabled
                voice_enabled = voice_output.value if hasattr(voice_output, 'value') else True
                if voice_enabled:
                    # Stop any ongoing speech before starting new one
                    stop_voice()
                    threading.Thread(target=_speak_text, args=(response,), daemon=True).start()
                
                return "", history, None
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                history = history or []
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": error_msg})
                return "", history, None

        # Connect the ask button
        ask_btn = gr.Button("Ask")
        ask_btn.click(
            fn=on_ask_click,
            inputs=[question, history_state, vector_store, model_choice],
            outputs=[question, chatbot, status]
        )
        
        # Connect the send button
        submit_btn.click(
            fn=on_ask_click,
            inputs=[msg, history_state, vector_store, model_choice],
            outputs=[msg, chatbot, status]
        )
        
        # Connect Enter key in the message input
        msg.submit(
            fn=on_ask_click,
            inputs=[msg, history_state, vector_store, model_choice],
            outputs=[msg, chatbot, status]
        )

    return demo


if __name__ == "__main__":
    import signal
    import sys
    
    def handle_sigint(signum, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, handle_sigint)
    
    try:
        demo = build_interface()
        # Set inbrowser=False to prevent opening a new browser tab
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=False,
            show_error=True,
            debug=True
        )
    except Exception as e:
        print(f"Error launching the app: {str(e)}")
        sys.exit(1)
