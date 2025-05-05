# 📚 MyStudyMate

MyStudyMate is an AI-powered study tool that helps students quickly convert lecture audio or PDF slides/textbooks into clean, structured notes and glossaries—perfect for review and exam prep.

It transcribes audio with Whisper, summarizes using GPT-4o, and exports everything in clean Markdown format. Just upload, wait, and learn.

## ✨ Features

- 🎙️ Audio transcription (MP3, WAV) via OpenAI Whisper
- 📄 PDF text extraction from slides, textbooks, or handouts
- 🧠 GPT-4o summarization into:
  - Hierarchical, bullet-point lecture notes
  - Glossary of key terms with definitions
- 💾 Downloadable outputs in Markdown and plain text
- 💸 Token and cost estimation before each summary (based on GPT-4o rates)
- 🔁 Regenerate summaries with a single click
- 💻 Streamlit UI with wide layout, loading indicators, and minimalist design

## 🚀 Getting Started

### 1. Clone the Repo

```
git clone https://github.com/YOUR_USERNAME/mystudymate.git
cd mystudymate
```

### 2. Set Up Your Environment

#### 📦 Dependencies

Install required packages (recommend using a virtual environment):

```
pip install -r requirements.txt
```

**Key dependencies:**

- streamlit
- whisper
- pydub
- openai
- PyMuPDF (fitz)
- tiktoken
- python-dotenv

#### 🔑 Environment Variables

Create a .env file in the root of the project with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

🧪 Make sure your key has access to GPT-4o and optionally Whisper if you're not using the local model.

## 🧠 How to Use

### 1. Start the app

```
streamlit run app.py
```

### 2. Upload your file

- Accepted formats:
  - .mp3 or .wav for lecture audio
  - .pdf for slides, textbooks, etc.

### 3. Transcription / Extraction

- Audio is chunked, transcribed with Whisper
- PDFs are parsed for raw text

### 4. View and Download

- Transcript or extracted text is shown in a scrollable view
- Download button available for saving the raw text

### 5. Summarize with GPT-4o

- Press the **✨ Generate Summary** button
- Estimated cost and token count is shown before generation
- Output includes:
  - `## Lecture Summary`: hierarchical bullet points
  - `## Glossary of Key Terms`: bolded terms and definitions
- Download the summary as Markdown

## 📊 Cost Estimation

MyStudyMate uses GPT-4o’s pricing:
| Token Type | Rate (USD) |
| ---------- | ------------ |
| Input | \$0.005 / 1K |
| Output | \$0.02 / 1K |
The app shows estimated cost before summarization using 900 output tokens as the default cap.

## 📁 Output Files

All outputs are saved in a local outputs/ folder:

- `{filename}_transcript.txt` – raw transcript (if audio)
- `{filename}_extracted.txt` – raw text (if PDF)
- `{filename}_summary.md` – formatted summary and glossary

## ⚙️ Optional: ffmpeg Setup

If Whisper fails to load audio properly, install ffmpeg:

```
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
```

## 🧼 Limitations

- GPT summaries are capped to ~50,000 characters input (~10K tokens)
- No paragraph/topic selection or deep interactivity (yet!)
- Requires internet connection for GPT summarization

## 📄 License

MIT License – free to use, modify, and build on.
