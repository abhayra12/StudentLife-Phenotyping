import pypdf
from pathlib import Path

def extract_text(pdf_path, output_path):
    print(f"Extracting text from {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text saved to {output_path}")

if __name__ == "__main__":
    pdf_path = Path("docs/studentlife.pdf")
    output_path = Path("docs/studentlife_paper.txt")
    extract_text(pdf_path, output_path)
