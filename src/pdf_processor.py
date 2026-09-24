# src/pdf_processor.py
import pypdf

class PDFProcessor:
    """Handles file operations and text extraction for uploaded documents."""
    
    @staticmethod
    def extract_text(pdf_file) -> str:
        """Reads an uploaded PDF file and returns the extracted text."""
        try:
            reader = pypdf.PdfReader(pdf_file)
            text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(text_pages)
        except Exception as e:
            raise ValueError(f"Failed to process PDF document: {e}")