import streamlit as st
import tempfile
import os
import whisper
import fitz  # PyMuPDF
from dotenv import load_dotenv
import openai
from openai import OpenAI
import tiktoken
from pydub import AudioSegment
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Output directory for all saved files
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Temp chunk directory for faster transcription
temp_chunks_dir = "temp_chunks"
os.makedirs(temp_chunks_dir, exist_ok=True)

def estimate_gpt4o_cost(text, model="gpt-4o", max_output_tokens=1800):
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(text))
    input_rate = 0.005 / 1000
    output_rate = 0.02 / 1000
    cost = (input_tokens * input_rate) + (max_output_tokens * output_rate)
    return input_tokens, max_output_tokens, cost

client = OpenAI()

def preprocess_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.apply_gain(-audio.max_dBFS)
    audio.export(output_path, format="wav")

def transcribe_audio_in_chunks(audio_path, model_size="tiny.en", chunk_ms=30_000):
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)

    total_chunks = (duration_ms + chunk_ms - 1) // chunk_ms
    completed_chunks = 0
    lock = threading.Lock()

    chunk_paths = []
    for i in range(total_chunks):
        start = i * chunk_ms
        chunk = audio[start:start + chunk_ms]
        chunk_filename = os.path.join(temp_chunks_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_filename, format="mp3")
        chunk_paths.append((i, chunk_filename))

    transcript_chunks = [None] * total_chunks
    failed_chunks = []

    def transcribe_chunk(idx, chunk_path):
        nonlocal completed_chunks
        try:
            thread_model = whisper.load_model(model_size)
            result = thread_model.transcribe(chunk_path)
            with lock:
                completed_chunks += 1
            return idx, result["text"]
        except Exception:
            with lock:
                failed_chunks.append((idx, chunk_path))
            return idx, None

    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1)) as executor:
        futures = {executor.submit(transcribe_chunk, idx, path): idx for idx, path in chunk_paths}

        while completed_chunks < total_chunks:
            with lock:
                progress = completed_chunks / total_chunks
            progress_bar.progress(progress)
            status_text.write(f"Transcribing chunk {completed_chunks} of {total_chunks}...")
            time.sleep(0.2)

        for future in as_completed(futures):
            idx, text = future.result()
            if text is not None:
                transcript_chunks[idx] = text

    # Serial retry for failed chunks
    if failed_chunks:
        status_text.write("Retrying failed chunks serially...")
        fallback_model = whisper.load_model(model_size)
        for idx, path in failed_chunks:
            try:
                result = fallback_model.transcribe(path)
                transcript_chunks[idx] = result["text"]
            except Exception:
                transcript_chunks[idx] = ""

    shutil.rmtree(temp_chunks_dir, ignore_errors=True)
    status_text.write("✅ Transcription complete.")
    progress_bar.progress(1.0)

    return " ".join([t or "" for t in transcript_chunks]).strip()

def generate_summary_and_glossary(input_text):
    prompt = (
        "You are a helpful academic assistant. A student has uploaded a transcript or textbook passage from a university lecture.\n\n"
        "**Your task has two parts:**\n\n"
        "1. Write a detailed and well-organized summary of the lecture in **hierarchical bullet-point format**, like a student might take in class. \n"
        "Use **main points**, then nest **sub-points**, and **sub-sub-points** where needed (using indented bullets). \n"
        "Capture not just key topics, but also examples, supporting ideas, and transitions. Think like you're building a study guide. \n"
        "Use Markdown for clarity, including headers or bolded section titles if useful.\n\n"
        "2. Then, write a **Glossary of Key Terms** as a separate section. For each term, provide a clear 1–2 sentence definition.\n\n"
        "Be thorough and explanatory, and use Markdown formatting throughout to keep it clean and readable.\n\n"
        f"{input_text[:12500]}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1800
    )
    usage = response.usage
    if usage and usage.total_tokens and usage.completion_tokens:
        if usage.completion_tokens >= 1800:
            st.warning("⚠️ The output may have been clipped. Try increasing `max_tokens` if needed.")
    return response.choices[0].message.content

def run_summary_generation(source_text, filename_prefix):
    summary_md = generate_summary_and_glossary(source_text)
    st.session_state.summary_md = summary_md
    st.session_state.last_summarized_file = filename_prefix
    summary_filename = f"{filename_prefix}_summary.md"
    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    st.session_state.summary_filename = summary_filename
    st.session_state.summary_path = summary_path

def display_summary_controls(uploaded_file_name, source_text):
    current_file_key = os.path.splitext(uploaded_file_name)[0]
    summary_already_generated = (
        st.session_state.get("summary_md") and
        st.session_state.get("last_summarized_file") == current_file_key and
        st.session_state.get("summary_path") and
        os.path.exists(st.session_state.get("summary_path"))
    )
    if not summary_already_generated:
        est_input, est_output, est_cost = estimate_gpt4o_cost(source_text)
        button_label = f"✨ Generate Summary and Glossary (est. ${est_cost:.2f}, {est_input} in / {est_output} out tokens)"
        if st.button(button_label, key="generate_button"):
            with st.spinner("Summarizing with GPT-4o..."):
                run_summary_generation(source_text, current_file_key)
    if summary_already_generated:
        st.markdown(st.session_state.summary_md)
        col1, col2 = st.columns([1, 1])
        with col1:
            with open(st.session_state.summary_path, "rb") as f:
                st.download_button("📄 Download Markdown", f, file_name=st.session_state.summary_filename, key="download_md")
        with col2:
            if st.button("🔁 Regenerate Summary", key="regenerate_button"):
                with st.spinner("Re-generating with GPT-4o..."):
                    run_summary_generation(source_text, current_file_key)

def load_existing_outputs(filename):
    base_name = os.path.splitext(filename)[0]
    
    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    extracted_path = os.path.join(output_dir, f"{base_name}_extracted.txt")
    summary_path = os.path.join(output_dir, f"{base_name}_summary.md")

    # Clear any previous session state
    st.session_state.pop("transcript", None)
    st.session_state.pop("transcript_path", None)
    st.session_state.pop("transcript_filename", None)

    st.session_state.pop("pdf_text", None)
    st.session_state.pop("pdf_text_path", None)
    st.session_state.pop("pdf_text_filename", None)

    st.session_state.pop("summary_md", None)
    st.session_state.pop("summary_path", None)
    st.session_state.pop("summary_filename", None)
    st.session_state.pop("last_summarized_file", None)

    # Load new ones if they exist
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            st.session_state.transcript = f.read()
        st.session_state.transcript_path = transcript_path
        st.session_state.transcript_filename = os.path.basename(transcript_path)

    if os.path.exists(extracted_path):
        with open(extracted_path, "r", encoding="utf-8") as f:
            st.session_state.pdf_text = f.read()
        st.session_state.pdf_text_path = extracted_path
        st.session_state.pdf_text_filename = os.path.basename(extracted_path)

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            st.session_state.summary_md = f.read()
        st.session_state.summary_path = summary_path
        st.session_state.summary_filename = os.path.basename(summary_path)
        st.session_state.last_summarized_file = base_name

st.title("MyStudyMate (MVP)")
st.write("Upload a lecture audio or PDF file")

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")
    file_extension = uploaded_file.name.split('.')[-1]
    base_name = os.path.splitext(uploaded_file.name)[0]
    load_existing_outputs(uploaded_file.name)
    if "summary_md" in st.session_state and "last_summarized_file" not in st.session_state:
        st.session_state.last_summarized_file = base_name
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name
    st.info(f"File saved to: {temp_file_path}")
    if file_extension in ["mp3", "wav"]:
        if "transcript" not in st.session_state:
            with st.spinner("Preprocessing audio and transcribing in chunks..."):
                preprocessed_path = temp_file_path.replace(".mp3", ".wav").replace(".wav", "_temp.wav")
                preprocess_audio(temp_file_path, preprocessed_path)
            transcript = transcribe_audio_in_chunks(preprocessed_path)
            st.session_state.transcript = transcript
            transcript_filename = f"{base_name}_transcript.txt"
            transcript_path = os.path.join(output_dir, transcript_filename)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            st.session_state.transcript_path = transcript_path
            st.session_state.transcript_filename = transcript_filename
        st.subheader("Transcript")
        st.text_area("Full Transcript", st.session_state.transcript, height=300)
        with open(st.session_state.transcript_path, "rb") as f:
            st.download_button("📄 Download Transcript", f, file_name=st.session_state.transcript_filename)
        source_text = st.session_state.get("transcript") or st.session_state.get("pdf_text")
        if source_text:
            display_summary_controls(uploaded_file.name, source_text)
    elif file_extension == "pdf":
        if "pdf_text" not in st.session_state:
            with st.spinner("Extracting text from PDF..."):
                text = ""
                with fitz.open(temp_file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                st.session_state.pdf_text = text
                pdf_text_filename = f"{base_name}_extracted.txt"
                pdf_text_path = os.path.join(output_dir, pdf_text_filename)
                with open(pdf_text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                st.session_state.pdf_text_path = pdf_text_path
                st.session_state.pdf_text_filename = pdf_text_filename
        st.subheader("Extracted Text")
        st.text_area("Full PDF Text", st.session_state.pdf_text, height=300)
        with open(st.session_state.pdf_text_path, "rb") as f:
            st.download_button("📄 Download Extracted Text", f, file_name=st.session_state.pdf_text_filename)
        source_text = st.session_state.get("transcript") or st.session_state.get("pdf_text")
        if source_text:
            display_summary_controls(uploaded_file.name, source_text)
