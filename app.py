import streamlit as st
import tempfile
import os
import whisper
import fitz  # PyMuPDF
from dotenv import load_dotenv
import openai
from openai import OpenAI
import tiktoken

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def estimate_gpt4o_cost(text, model="gpt-4o", max_output_tokens=1800):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(text))

    # Pricing (May 2025)
    input_rate = 0.005 / 1000   # $5.00 per 1M input tokens
    output_rate = 0.02 / 1000   # $20.00 per 1M output tokens

    cost = (input_tokens * input_rate) + (max_output_tokens * output_rate)
    return input_tokens, max_output_tokens, cost

client = OpenAI()

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
        max_tokens=1800  # Allows for more detailed output
    )

    # Fallback warning if output token limit might be hit
    usage = response.usage
    if usage and usage.total_tokens and usage.completion_tokens:
        if usage.completion_tokens >= 1800:
            st.warning("⚠️ The output may have been clipped. Try increasing `max_tokens` if needed.")

    return response.choices[0].message.content

def run_summary_generation(source_text, filename_prefix):
    summary_md = generate_summary_and_glossary(source_text)
    st.session_state.summary_md = summary_md
    st.session_state.last_summarized_file = filename_prefix  # Track the file

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
        st.session_state.get("last_summarized_file") == current_file_key
    )

    if not summary_already_generated:
        est_input, est_output, est_cost = estimate_gpt4o_cost(source_text)
        button_label = f"✨ Generate Summary and Glossary (est. ${est_cost:.2f}, {est_input} in / {est_output} out tokens)"
        if st.button(button_label, key="generate_button"):
            with st.spinner("Summarizing with GPT-4o..."):
                run_summary_generation(source_text, current_file_key)

    if summary_already_generated:
        st.subheader("📘 Summary + Glossary")
        st.markdown(st.session_state.summary_md)

        col1, col2 = st.columns([1, 1])
        with col1:
            with open(st.session_state.summary_path, "rb") as f:
                st.download_button("📄 Download Markdown", f, file_name=st.session_state.summary_filename, key="download_md")

        with col2:
            if st.button("🔁 Regenerate Summary", key="regenerate_button"):
                with st.spinner("Re-generating with GPT-4o..."):
                    run_summary_generation(source_text, current_file_key)


# Output directory for all saved files
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

st.title("MyStudyMate (MVP)")
st.write("Upload a lecture audio or PDF file")

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")
    file_extension = uploaded_file.name.split('.')[-1]

    # Save file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.info(f"File saved to: {temp_file_path}")

    # Handle audio transcription
    if file_extension in ["mp3", "wav"]:
        if "transcript" not in st.session_state:
            with st.spinner("Transcribing..."):
                model = whisper.load_model("tiny")
                result = model.transcribe(temp_file_path)
                st.session_state.transcript = result["text"]

                # Save transcript to a file
                transcript_filename = os.path.splitext(uploaded_file.name)[0] + "_transcript.txt"
                transcript_path = os.path.join(output_dir, transcript_filename)
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(st.session_state.transcript)

                st.session_state.transcript_path = transcript_path
                st.session_state.transcript_filename = transcript_filename

        st.subheader("Transcript")
        st.text_area("Full Transcript", st.session_state.transcript, height=300)

        with open(st.session_state.transcript_path, "rb") as f:
            st.download_button("📄 Download Transcript", f, file_name=st.session_state.transcript_filename)

        # After transcript or extracted text is displayed
        source_text = st.session_state.get("transcript") or st.session_state.get("pdf_text")
        if source_text:
            display_summary_controls(uploaded_file.name, source_text)


    # Handle PDF text extraction
    elif file_extension == "pdf":
        if "pdf_text" not in st.session_state:
            with st.spinner("Extracting text from PDF..."):
                text = ""
                with fitz.open(temp_file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                st.session_state.pdf_text = text

                # Save extracted text
                pdf_text_filename = os.path.splitext(uploaded_file.name)[0] + "_extracted.txt"
                pdf_text_path = os.path.join(output_dir, pdf_text_filename)
                with open(pdf_text_path, "w", encoding="utf-8") as f:
                    f.write(text)

                st.session_state.pdf_text_path = pdf_text_path
                st.session_state.pdf_text_filename = pdf_text_filename

        st.subheader("Extracted Text")
        st.text_area("Full PDF Text", st.session_state.pdf_text, height=300)

        with open(st.session_state.pdf_text_path, "rb") as f:
            st.download_button("📄 Download Extracted Text", f, file_name=st.session_state.pdf_text_filename)

        # After transcript or extracted text is displayed
        source_text = st.session_state.get("transcript") or st.session_state.get("pdf_text")
        if source_text:
            display_summary_controls(uploaded_file.name, source_text)
