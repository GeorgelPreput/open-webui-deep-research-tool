import asyncio
import logging

from deep_research.config.constants import HANDLE_PDFS, PDF_MAX_PAGES
from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.web.pdf_extract")


async def extract_text_from_pdf(ctx: RunContext, pdf_content) -> str:
    """Fallback PDF text extractor (PyPDF2 → pdfplumber).

    Rescue path used by _extract_pdf_with_primary_fallback when
    Open WebUI's document extraction Loader is unavailable or
    returns unusable output. Honours HANDLE_PDFS and
    PDF_MAX_PAGES and runs the CPU-bound work in the shared
    thread-pool executor.
    """
    if not HANDLE_PDFS:
        return "PDF processing is disabled in settings."

    # Ensure we have bytes for the PDF content
    if isinstance(pdf_content, str):
        if pdf_content.startswith("%PDF"):
            pdf_content = pdf_content.encode("utf-8", errors="ignore")
        else:
            return "Error: Invalid PDF content format"

    # Limit extraction to configured max pages to avoid too much processing
    max_pages = PDF_MAX_PAGES

    try:
        # Try PyPDF2 first
        try:
            import io

            from PyPDF2 import PdfReader

            # Use ThreadPoolExecutor for CPU-intensive PDF processing
            def extract_with_pypdf():
                try:
                    # Create a reader object
                    pdf_file = io.BytesIO(pdf_content)
                    pdf_reader = PdfReader(pdf_file)

                    # Get the total number of pages
                    num_pages = len(pdf_reader.pages)
                    logger.info(
                        f"PDF has {num_pages} pages, extracting up to {max_pages}"
                    )

                    # Extract text from each page up to the limit
                    text = []
                    for page_num in range(min(num_pages, max_pages)):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                text.append(f"Page {page_num + 1}:\n{page_text}")
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num}: {e}")

                    # Join all pages with spacing
                    full_text = "\n\n".join(text)

                    # Add a note if we limited the page count
                    if num_pages > max_pages:
                        full_text += f"\n\n[Note: This PDF has {num_pages} pages, but only the first {max_pages} were processed.]"

                    return full_text if full_text.strip() else None
                except Exception as e:
                    logger.error(f"Error in PDF extraction with PyPDF2: {e}")
                    return None

            # Execute in thread pool
            loop = asyncio.get_running_loop()
            pdf_extract_task = loop.run_in_executor(
                ctx.executor, extract_with_pypdf
            )
            full_text = await pdf_extract_task

            if full_text and full_text.strip():
                logger.info(
                    f"Successfully extracted text from PDF using PyPDF2: {len(full_text)} chars"
                )
                return full_text
            else:
                logger.warning(
                    "PyPDF2 extraction returned empty text, trying pdfplumber..."
                )
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}, trying pdfplumber...")

        # Try pdfplumber as a fallback
        try:
            import io

            import pdfplumber

            # Use ThreadPoolExecutor for CPU-intensive PDF processing
            def extract_with_pdfplumber():
                try:
                    pdf_file = io.BytesIO(pdf_content)
                    with pdfplumber.open(pdf_file) as pdf:
                        # Get total pages
                        num_pages = len(pdf.pages)

                        text = []
                        for i, page in enumerate(pdf.pages[:max_pages]):
                            try:
                                page_text = page.extract_text() or ""
                                if page_text.strip():
                                    text.append(f"Page {i + 1}:\n{page_text}")
                            except Exception as page_error:
                                logger.warning(
                                    f"Error extracting page {i} with pdfplumber: {page_error}"
                                )

                        full_text = "\n\n".join(text)

                        # Add a note if we limited the page count
                        if num_pages > max_pages:
                            full_text += f"\n\n[Note: This PDF has {num_pages} pages, but only the first {max_pages} were processed.]"

                        return full_text
                except Exception as e:
                    logger.error(f"Error in PDF extraction with pdfplumber: {e}")
                    return None

            # Execute in thread pool
            loop = asyncio.get_running_loop()
            pdf_extract_task = loop.run_in_executor(
                ctx.executor, extract_with_pdfplumber
            )
            full_text = await pdf_extract_task

            if full_text and full_text.strip():
                logger.info(
                    f"Successfully extracted text from PDF using pdfplumber: {len(full_text)} chars"
                )
                return full_text
            else:
                logger.warning("pdfplumber extraction returned empty text")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")

        # If both methods failed but we can tell it's a PDF, provide a more useful message
        if pdf_content.startswith(b"%PDF"):
            logger.warning(
                "PDF detected but text extraction failed. May be scanned or encrypted."
            )
            return "This appears to be a PDF document, but text extraction failed. The PDF may contain scanned images rather than text, or it may be encrypted/protected."

        return "Could not extract text from PDF. The file may not be a valid PDF or may contain security restrictions."

    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return f"Error extracting text from PDF: {str(e)}"
