import streamlit as st
import tempfile
import os
import whisper

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

    # If it's audio, run transcription
    if file_extension in ["mp3", "wav"]:
        if "transcript" not in st.session_state:
            with st.spinner("Transcribing..."):
                model = whisper.load_model("tiny")
                result = model.transcribe(temp_file_path)
                st.session_state.transcript = result["text"]

            # Save transcript to a file
            transcript_filename = os.path.splitext(uploaded_file.name)[0] + "_transcript.txt"
            transcript_path = os.path.join(os.getcwd(), transcript_filename)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(st.session_state.transcript)

            st.session_state.transcript_path = transcript_path
            st.session_state.transcript_filename = transcript_filename

        # Show transcript
        st.subheader("Transcript")
        st.text_area("Full Transcript", st.session_state.transcript, height=300)

        # Download button
        with open(st.session_state.transcript_path, "rb") as f:
            st.download_button("📄 Download Transcript", f, file_name=st.session_state.transcript_filename)