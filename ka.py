import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import easyocr
import io
import tempfile
import os
from save__json import save_text_to_json   # JSON SAVE IMPORT

reader = easyocr.Reader(["en"], gpu=False)

st.title("📄 Universal OCR App (PyMuPDF Version)")
st.write("Extract text from Scanned PDFs, Image-based PDFs, Images, and CSV files — No Poppler Needed!")


# ---------- CSV ----------
def extract_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string()


# ---------- Image OCR ----------
def extract_pdf(file_bytes):
    """
    Smarter PDF extractor:
    1) Try direct text extraction (for digital PDFs).
    2) If page text is empty/too small, fallback to OCR (for scanned pages).
    """
    full_text = []

    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index in range(len(pdf)):
        page = pdf[page_index]

        # 1) Try direct text extraction
        page_text = page.get_text("text").strip()

        # Heuristic: if we get enough characters, trust direct text
        if len(page_text) > 50:
            full_text.append(page_text)
            continue

        # 2) Fallback to OCR for scanned/empty pages
        pix = page.get_pixmap(dpi=120)  # lower dpi to avoid RAM issues

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_file:
            img_file.write(pix.tobytes())
            img_path = img_file.name

        ocr_result = reader.readtext(img_path, detail=0)
        os.remove(img_path)

        ocr_text = " ".join(ocr_result).strip()
        full_text.append(ocr_text)

    return "\n\n".join(full_text)


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload PDF / CSV / Image",
    type=["pdf", "csv", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    ext = uploaded_file.name.split(".")[-1].lower()

    st.write(f"### File Type: {ext.upper()}")


    # ---------- CSV ----------
    if ext == "csv":
        out = extract_csv(file_bytes)
        st.text_area("Extracted Text", out, height=500)

        # Save JSON
        saved_file = save_text_to_json(out)
        st.success(f"JSON saved to: {saved_file}")
        # 🔥 Automatically build vectors
        import subprocess
        subprocess.Popen([r"C:\Users\Academytraining\Desktop\ocr\ding\Scripts\python.exe", "vect.py", "--build"])

        # Download button
        with open(saved_file, "rb") as f:
            st.download_button(
                label="📥 Download JSON File",
                data=f,
                file_name="extracted_text.json",
                mime="application/json"
            )


    # ---------- PDF ----------
    elif ext == "pdf":
        st.info("🔍 Extracting from PDF using PyMuPDF + OCR...")
        out = extract_pdf(file_bytes)
        st.text_area("Extracted Text (PDF OCR)", out, height=500)

        # Save JSON
        saved_file = save_text_to_json(out)
        st.success(f"JSON saved to: {saved_file}")
        # vectorizeautomatically
        import subprocess
        subprocess.Popen([r"C:\Users\Academytraining\Desktop\ocr\ding\Scripts\python.exe", "vect.py", "--build"])


        # Download button
        with open(saved_file, "rb") as f:
            st.download_button(
                label="📥 Download JSON File",
                data=f,
                file_name="extracted_text.json",
                mime="application/json"
            )


    # ---------- IMAGE ----------
    else:
        st.info("📸 Extracting from Image...")
        out = extract_image(file_bytes)
        st.text_area("Extracted Text (Image OCR)", out, height=500)

        # Save JSON
        saved_file = save_text_to_json(out)
        st.success(f"JSON saved to: {saved_file}")
        # vectorizeautomatically
        import subprocess
        subprocess.Popen([r"C:\Users\Academytraining\Desktop\ocr\ding\Scripts\python.exe", "vect.py", "--build"])

        # Download button
        with open(saved_file, "rb") as f:
            st.download_button(
                label="📥 Download JSON File",
                data=f,
                file_name="extracted_text.json",
                mime="application/json"
            )
