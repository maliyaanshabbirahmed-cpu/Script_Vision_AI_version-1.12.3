import io
import json
import os
import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import google.generativeai as genai

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

st.set_page_config(page_title="Text Detector | Image to Excel", page_icon="📄", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
:root { --gold: #d6ad55; }
.stApp { background: radial-gradient(circle at 15% 0%, #202020 0, #101010 42%, #090909 100%); color: #f4f0e7; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background: #111; border-right: 1px solid #3d3320; }
h1,h2,h3 { color: #f3d58b !important; }
div[data-testid="stMetric"] { background: #171717; border: 1px solid #3d3320; padding: 14px; border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border: 1px solid #c49a45; background: #c49a45; color: #101010; font-weight: 700; border-radius: 9px; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #f3d58b; background: #f3d58b; color: #000; }
div[data-testid="stFileUploader"] { background: #151515; border: 1px dashed #8d7138; border-radius: 12px; padding: 10px; }
.small-note { color: #bdb6a8; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

st.title("TEXT DETECTOR")
st.caption("AI-powered image and document extraction • Editable results • CSV and Excel export")

# ---------- Configuration ----------
def get_api_key() -> str:
    try:
        value = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        value = ""
    return value or os.getenv("GEMINI_API_KEY", "")

def image_to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()

def pdf_to_images(data: bytes, dpi: int = 180) -> list[Image.Image]:
    if fitz is None:
        raise RuntimeError("PDF support requires PyMuPDF. Install dependencies from requirements.txt.")
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return pages

def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return valid JSON. Try again or use a clearer image.")
    result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Unexpected response format from the model.")
    return result

def call_vision_model(image: Image.Image, api_key: str, model_name: str, language_hint: str, task_hint: str) -> dict[str, Any]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
You are a high-accuracy document transcription and table-structure extraction engine.
Read the uploaded image carefully. Preserve the source faithfully; do not translate, paraphrase,
correct spelling, normalize numbers, invent missing content, or silently repair grammar.
Language/script hint: {language_hint or 'Automatically identify all visible languages and scripts'}.
Document hint: {task_hint or 'Determine the document type from the image'}.

Return ONLY valid JSON with this exact schema:
{{
  "document_type": "short description",
  "detected_languages": ["language/script"],
  "full_text": "faithful transcription with line breaks represented as \\n",
  "tables": [
    {{
      "title": "table title or empty string",
      "headers": ["column 1", "column 2"],
      "rows": [["cell 1", "cell 2"], ["cell 1", "cell 2"]]
    }}
  ],
  "other_lines": ["non-tabular line 1", "non-tabular line 2"],
  "uncertain_regions": [
    {{"text_or_location": "brief location or fragment", "reason": "why uncertain"}}
  ]
}}

Rules:
- Preserve original spelling, capitalization, punctuation, symbols, dates, decimal places and digit shapes as closely as possible.
- Preserve row/column order and blank cells in tables. Never merge separate columns.
- If no table exists, return an empty tables array.
- If a character cannot be read, use [unclear] and describe it in uncertain_regions.
- Do not guess illegible words. Do not add explanatory commentary.
- For handwriting, inspect each word and character; cursive and unusual writing may be uncertain.
- Keep Urdu/Arabic/right-to-left text in its original script. Do not transliterate unless it is visibly transliterated.
"""
    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": image_to_png_bytes(image)}],
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    return extract_json_payload(response.text)

def rows_to_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    tables = result.get("tables") or []
    normalized = []
    for table_index, table in enumerate(tables, start=1):
        headers = [str(x) for x in (table.get("headers") or [])]
        rows = table.get("rows") or []
        max_cols = max([len(headers)] + [len(r) for r in rows] + [0])
        if max_cols == 0:
            continue
        if not headers:
            headers = [f"Column {i+1}" for i in range(max_cols)]
        elif len(headers) < max_cols:
            headers += [f"Column {i+1}" for i in range(len(headers), max_cols)]
        for row in rows:
            row = ["" if x is None else str(x) for x in row]
            row += [""] * (max_cols - len(row))
            normalized.append({"Table": table_index, **dict(zip(headers, row[:max_cols]))})
    if normalized:
        return pd.DataFrame(normalized)
    lines = result.get("other_lines") or result.get("full_text", "").splitlines()
    return pd.DataFrame({"Extracted text": [str(x) for x in lines]})

def make_excel(result: dict[str, Any], dataframe: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="Extracted Data", index=False)
        text_df = pd.DataFrame({"Line": (result.get("full_text") or "").splitlines()})
        text_df.to_excel(writer, sheet_name="Full Text", index=False)
        uncertain = result.get("uncertain_regions") or []
        pd.DataFrame(uncertain if uncertain else [{"text_or_location": "", "reason": ""}]).to_excel(
            writer, sheet_name="Review Notes", index=False
        )
    output.seek(0)
    return output.getvalue()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    api_key = get_api_key()
    manual_key = st.text_input("Gemini API key", type="password", help="Stored only for this session unless you add it to Streamlit Secrets.")
    api_key = manual_key.strip() or api_key
    model_name = st.text_input("Vision model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    language_hint = st.selectbox("Language / script", ["Auto detect", "English", "Urdu", "Arabic", "English + Urdu", "Other / mixed"])
    task_hint = st.selectbox("Document type", ["Auto detect", "Handwritten notes", "Printed page", "Table / register", "Form / invoice", "Book / textbook"])
    st.markdown("---")
    st.markdown("**Accuracy note**")
    st.markdown('<p class="small-note">No OCR model can guarantee perfect recognition for every handwriting style, image quality or language. Review uncertain words before exporting.</p>', unsafe_allow_html=True)

# ---------- Upload and crop ----------
uploaded = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"])
if uploaded is None:
    st.info("Upload a clear image or PDF to begin. You can crop an image before extraction.")
    st.markdown("**Workflow:** Upload → Crop / select PDF page → AI transcription → Review and edit → Download CSV / Excel.")
    st.stop()

raw = uploaded.getvalue()
is_pdf = uploaded.name.lower().endswith(".pdf")
try:
    if is_pdf:
        pages = pdf_to_images(raw)
        if not pages:
            st.error("The PDF contains no readable pages.")
            st.stop()
        page_number = st.selectbox("Choose PDF page", options=list(range(1, len(pages) + 1)))
        source_image = pages[page_number - 1]
    else:
        source_image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
except Exception as exc:
    st.error(f"Could not open this file: {exc}")
    st.stop()

st.subheader("1. Preview and crop")
left, right = st.columns([1.1, 1])
with left:
    st.image(source_image, caption="Original upload", use_container_width=True)
with right:
    width, height = source_image.size
    st.caption(f"Image dimensions: {width} × {height} pixels")
    crop_enabled = st.checkbox("Enable custom crop", value=False)
    if crop_enabled:
        st.caption("Set crop boundaries. The preview and OCR use the cropped region.")
        x1 = st.slider("Left", 0, max(0, width - 1), 0)
        x2 = st.slider("Right", 1, width, width)
        y1 = st.slider("Top", 0, max(0, height - 1), 0)
        y2 = st.slider("Bottom", 1, height, height)
        if x2 <= x1 or y2 <= y1:
            st.warning("Crop boundaries are invalid. Adjust the sliders.")
            cropped = source_image
            crop_valid = False
        else:
            cropped = source_image.crop((x1, y1, x2, y2))
            crop_valid = True
    else:
        cropped = source_image
        crop_valid = True
    st.image(cropped, caption="Region sent to the AI model", use_container_width=True)

st.subheader("2. Extract text and tables")
if st.button("Run AI text detection", type="primary", disabled=not crop_valid, use_container_width=True):
    if not api_key:
        st.error("Add a Gemini API key in the sidebar or configure GEMINI_API_KEY in Streamlit Secrets.")
    else:
        with st.spinner("Analyzing the document. Handwriting may take longer..."):
            try:
                hint = "" if language_hint == "Auto detect" else language_hint
                doc_hint = "" if task_hint == "Auto detect" else task_hint
                result = call_vision_model(cropped, api_key, model_name.strip(), hint, doc_hint)
                st.session_state["ocr_result"] = result
                st.session_state["ocr_filename"] = uploaded.name
                st.session_state["ocr_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["ocr_dataframe"] = rows_to_dataframe(result)
                st.success("Extraction completed. Review the transcription before exporting.")
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                st.caption("Check the API key, model name, internet connection, quota and image quality.")

# ---------- Review and export ----------
result = st.session_state.get("ocr_result")
if result:
    st.subheader("3. Review extracted content")
    st.caption(f"File: {st.session_state.get('ocr_filename', '')} · Processed: {st.session_state.get('ocr_timestamp', '')}")
    st.text_area("Full transcription (edit before exporting)", value=result.get("full_text", ""), height=220, key="edited_full_text")
    if result.get("uncertain_regions"):
        st.warning("The model identified content that needs manual review.")
        st.dataframe(pd.DataFrame(result["uncertain_regions"]), use_container_width=True, hide_index=True)
    tables = result.get("tables") or []
    if tables:
        for i, table in enumerate(tables):
            st.markdown(f"**Table {i+1}: {table.get('title') or 'Extracted table'}**")
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if not headers and rows:
                headers = [f"Column {j+1}" for j in range(max(len(r) for r in rows))]
            max_cols = max([len(headers)] + [len(r) for r in rows] + [0])
            headers = list(headers) + [f"Column {j+1}" for j in range(len(headers), max_cols)]
            safe_rows = [list(r) + [""] * (max_cols - len(r)) for r in rows]
            if max_cols:
                editable = pd.DataFrame(safe_rows, columns=headers[:max_cols])
                st.data_editor(editable, key=f"table_editor_{i}", num_rows="dynamic", use_container_width=True)
    default_df = st.session_state.get("ocr_dataframe", pd.DataFrame())
    st.markdown("**Export data (editable grid)**")
    edited_df = st.data_editor(default_df, key="export_data_editor", num_rows="dynamic", use_container_width=True)
    csv_bytes = edited_df.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = make_excel({"full_text": st.session_state.get("edited_full_text", result.get("full_text", "")),
                              "uncertain_regions": result.get("uncertain_regions", [])}, edited_df)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download CSV", data=csv_bytes, file_name="extracted_data.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Download Excel", data=excel_bytes, file_name="extracted_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    st.caption("CSV/Excel contain extracted structured data. The Full Text sheet preserves the model's line-by-line transcription for comparison.")
