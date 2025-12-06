import os
import re
import signal
import tempfile
import math
import threading
import queue
import time
import atexit
from typing import Optional, List, Dict, Any, Union, Tuple
import sounddevice as sd
import numpy as np
import soundfile as sf
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


class VoiceController:
    """Thread-safe controller for voice operations."""
    
    def __init__(self):
        self.engine = None
        self.stop_speaking = False
        self.lock = threading.Lock()
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        self._initialize_engine()

    def _initialize_engine(self) -> bool:
        """Safely initialize the TTS engine."""
        if self.engine is not None:
            return True
            
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            return True
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            self.engine = None
            return False

    def stop_voice(self) -> None:
        """Stop any ongoing speech and clean up resources."""
        with self.lock:
            self.stop_speaking = True
            if self.engine:
                try:
                    self.engine.stop()
                except Exception as e:
                    print(f"Error stopping TTS engine: {e}")
                self.engine = None

    def _process_queue(self) -> None:
        """Process the speech queue in a separate thread."""
        while True:
            try:
                text = self.audio_queue.get(timeout=1.0)
                if text is None:  # Shutdown signal
                    break
                    
                self._speak_text(text)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech processing: {e}")

    def _speak_text(self, text: str) -> None:
        """Internal method to speak text with proper error handling."""
        if not text.strip():
            return

        with self.lock:
            self.stop_speaking = False
            if not self._initialize_engine():
                return

        try:
            # Split text into sentences for better interruption handling
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                with self.lock:
                    if self.stop_speaking or self.engine is None:
                        break
                
                if sentence.strip():  # Skip empty sentences
                    self.engine.say(sentence)
                    self.engine.runAndWait()
                    
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            with self.lock:
                if self.stop_speaking and self.engine:
                    try:
                        self.engine.stop()
                    except Exception as e:
                        print(f"Error stopping TTS engine: {e}")
                    self.engine = None

    def speak(self, text: str, block: bool = False) -> None:
        """Add text to the speech queue.
        
        Args:
            text: The text to speak
            block: If True, wait until speech is finished
        """
        if not text.strip():
            return

        self.audio_queue.put(text)
        
        # Start processing thread if not already running
        with self.lock:
            if not self.is_processing:
                self.is_processing = True
                self.processing_thread = threading.Thread(
                    target=self._process_queue,
                    daemon=True
                )
                self.processing_thread.start()
        
        if block:
            self.audio_queue.join()

    def cleanup(self) -> None:
        """Clean up resources and stop all threads."""
        self.stop_voice()
        self.audio_queue.put(None)  # Signal thread to exit
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

# Global voice controller instance
voice_controller = VoiceController()

def stop_voice() -> str:
    """Stop any ongoing voice output."""
    voice_controller.stop_voice()
    return ""

def _speak_text(text: str) -> None:
    """Speak the given text using the voice controller."""
    voice_controller.speak(text)

def cleanup_resources():
    """Clean up resources before exiting."""
    voice_controller.cleanup()

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
    """Record audio from the microphone with better error handling."""
    try:
        # List available audio devices
        devices = sd.query_devices()
        default_input = sd.default.device[0]  # Get default input device
        print(f"Using audio device: {devices[default_input]['name']}")

        # Record with error handling
        print("Recording... Speak now!")
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype='float32',
            device=default_input
        )
        sd.wait()  # Wait until recording is finished
        print("Recording finished")
        return recording
    except Exception as e:
        print(f"Error recording audio: {e}")
        # Try to list available devices for debugging
        try:
            print("\nAvailable audio devices:")
            for i, device in enumerate(sd.query_devices()):
                print(f"{i}: {device['name']} (Inputs: {device['max_input_channels']})")
        except:
            print("Could not list audio devices")
        return None


def listen_microphone() -> str:
    """Listen to microphone and return the recognized text using sounddevice for recording."""
    try:
        # Record audio
        audio_data = record_audio()
        if audio_data is None:
            return "Error: Could not record audio"

        # Save to temporary file
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, 44100)

                # Use speech recognition
                r = sr.Recognizer()
                with sr.AudioFile(tmp_file.name) as source:
                    print("Processing audio...")
                    audio = r.record(source)

                try:
                    print("Transcribing speech...")
                    text = r.recognize_google(audio)
                    print(f"Recognized: {text}")
                    return text
                except sr.UnknownValueError:
                    return "Could not understand audio"
                except sr.RequestError as e:
                    return f"Could not request results; {e}"
                except Exception as e:
                    return f"Recognition error: {str(e)}"

        finally:
            # Clean up temporary file
            if tmp_file and os.path.exists(tmp_file.name):
                try:
                    os.unlink(tmp_file.name)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {e}")

    except Exception as e:
        error_msg = f"Error in voice recognition: {str(e)}"
        print(error_msg)
        return error_msg


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
                    " Upload Document",
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
                    voice_input_btn = gr.Button(" Voice Input")

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


def handle_sigint(signum, frame):
    """Handle interrupt signals for clean shutdown."""
    print("\nShutting down gracefully...")
    cleanup_resources()
    sys.exit(0)


if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        # Build and launch the interface
        interface = build_interface()
        interface.launch(share=False, debug=False, show_error=True)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching the app: {str(e)}")
        sys.exit(1)
