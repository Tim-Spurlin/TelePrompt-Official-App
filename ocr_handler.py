import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Use the tesseract folder inside the current project folder.
tesseract_exe_path = os.path.join(os.path.dirname(__file__), "tesseract", "tesseract.exe")
tessdata_dir_path = os.path.join(os.path.dirname(__file__), "tesseract", "tessdata")
tessdata_dir_config = '--tessdata-dir "' + tessdata_dir_path + '"'
pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path
pytesseract.pytesseract.tessdata_dir_config = tessdata_dir_config

print(f"Tesseract exe path: {tesseract_exe_path}")
print(f"Current Working Directory (ocr_handler.py): {os.getcwd()}")

def extract_text(file_path):
    """
    Always uses Tesseract for OCR, ignoring any choice.
    """
    file_lower = file_path.lower()
    if file_lower.endswith(".pdf"):
        return extract_text_from_pdf_tesseract(file_path)
    else:
        return extract_text_tesseract(file_path)

def extract_text_from_image_tesseract(image_path):
    """
    Runs OCR on an image file using Tesseract.
    """
    try:
        image = Image.open(image_path)
        text_from_ocr = pytesseract.image_to_string(image)
        print(f"Extracted text from Tesseract (first 100 chars): {text_from_ocr[:100]}...")  # Debug Print
        return text_from_ocr.strip()
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        return ""

def extract_text_from_pdf_tesseract(pdf_path):
    """
    Converts a PDF to images and runs Tesseract OCR on each page.
    Uses Poppler binaries bundled with the application.
    """
    import os
    # Set the poppler_path to the bundled binaries folder.
    # This path is relative to ocr_handler.py. Since ocr_handler.py is in the project root,
    # and the poppler folder is in the project root as well:
    poppler_path = os.path.join(os.path.dirname(__file__), "poppler", "bin")
    
    # Convert PDF pages to images using the bundled poppler_path.
    images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
    extracted_text = ""
    for i, img in enumerate(images):
        img_path = f"temp_page_{i}.png"
        img.save(img_path, "PNG")
        extracted_text += extract_text_from_image_tesseract(img_path) + "\n"
        os.remove(img_path)
    return extracted_text.strip()

def extract_text_tesseract(file_path):
    """
    Detects file type and extracts text using Tesseract.
    """
    file_ext = file_path.lower().split('.')[-1]
    if file_ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        return extract_text_from_image_tesseract(file_path)
    elif file_ext == "pdf":
        return extract_text_from_pdf_tesseract(file_path)
    else:
        return "Unsupported file type for Tesseract OCR."

def extract_text_from_txt(txt_path):
    """
    Reads a .txt file and returns its text as a single string.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file '{txt_path}': {e}")
        return ""

