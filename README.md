# MyStudyMate (MVP)

**MyStudyMate** is an AI-powered tool that helps students turn lecture recordings and slides into structured study materials. Upload a lecture audio file or PDF, and get a clean summary and glossary in return.

This project is in its MVP stage and currently supports:

- ✅ File upload for `.mp3`, `.wav`, and `.pdf`
- ✅ Temporary saving of uploaded files
- ✅ Automatic transcription
- 🚧 (Coming soon) Automatic summarization
- 🚧 (Coming soon) Flashcard generation and chatbot support

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/my-study-mate.git
   cd my-study-mate
   ```
2. Set up your environment:
   ```python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```

## 🔧 Tech Stack

- Python 3
- Streamlit
- (Planned) OpenAI Whisper
- (Planned) GPT-based summarization

## 📂 Project Vision

- This is the foundation for a personal study assistant that:
- Lets you focus on listening instead of writing notes
- Builds summaries, glossaries, and flashcards for deep review
- Gives you a chatbot to clarify material, all from your own lectures

## 🙌 Future Goals

- Audio transcription with Whisper
- Summarization and glossary extraction with GPT-4
- Flashcard generation (for Anki or web)
- Organized by course and session

## 📄 License

MIT License – feel free to fork, extend, or contribute.
