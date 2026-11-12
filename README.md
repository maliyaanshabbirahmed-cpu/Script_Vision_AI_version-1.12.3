# Text Detector — Image to CSV / Excel

A Streamlit application that uses a multimodal Gemini vision model to transcribe printed or handwritten text from images and PDF pages, extract table structure, allow manual review, and export CSV / Excel files.

## Features
- PNG, JPG, JPEG, WEBP, BMP, TIFF and PDF page input
- Manual crop boundaries before OCR
- Vision-model transcription with a low-temperature, source-faithful prompt
- Attempts to preserve original spelling, punctuation, numbers, line breaks and table structure
- Uncertain-region reporting
- Editable transcription and extracted data
- CSV and Excel exports; Excel includes Extracted Data, Full Text and Review Notes sheets
- Dark black-and-gold interface

## Requirements
- Python 3.10–3.12 recommended
- A Google Gemini API key with access to the configured vision model
- Internet access while extracting text

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enter your Gemini API key in the sidebar, or create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example`.

## Deploy on Streamlit Community Cloud
1. Upload the project files to a GitHub repository.
2. Set the main file path to `app.py`.
3. Add `GEMINI_API_KEY` under Streamlit app Settings → Secrets.
4. Deploy.

## Accuracy and limitations
No current OCR/vision model can guarantee perfect recognition of every handwriting style, cursive script, language, low-resolution image, or overlapping mark. The model may misread characters or table boundaries. Always inspect the editable result and the Review Notes before using exported data. For critical records, compare the transcription against the original upload.

The application does not train a custom handwriting model. It sends the selected image region to the configured Gemini vision model for inference. API usage, availability, pricing, data handling and rate limits are controlled by Google and your account settings.

## Configuration
- `GEMINI_API_KEY`: API key (Streamlit Secrets or environment variable)
- `GEMINI_MODEL`: optional model override; defaults to `gemini-2.5-flash`

Never commit API keys or `.streamlit/secrets.toml` to GitHub.
