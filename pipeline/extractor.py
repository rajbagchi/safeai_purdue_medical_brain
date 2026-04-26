"""
Multi-pass PDF extraction engine.
"""

import os
import json
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

import fitz  # PyMuPDF
import numpy as np
from rapidfuzz import fuzz

from .config import ExtractionConfig

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import tabulate  # noqa: F401 — pandas uses for DataFrame.to_markdown
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


def _dataframe_to_markdown(df) -> str:
    """Prefer pandas to_markdown (needs tabulate); fall back to CSV-like text."""
    if hasattr(df, "to_markdown"):
        try:
            return df.to_markdown(index=False)
        except Exception:
            pass
    return df.to_string(index=False)


class MultiPassExtractor:
    """
    Intelligent multi-pass extraction engine.
    Adapts strategy based on document content.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.passes: List[Dict] = []
        self.document_profile: Dict = {}

        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "validation"), exist_ok=True)

    def analyze_document(self) -> Dict:
        """Quick first pass to understand document structure."""
        print("\n🔍 [Pass 0] Analyzing document structure...")

        doc = fitz.open(self.config.pdf_path)

        profile: Dict[str, Any] = {
            "total_pages": len(doc),
            "has_tables": False,
            "has_images": False,
            "has_scanned_pages": False,
            "page_types": [],
            "estimated_tables": 0,
            "language": "unknown",
            "page_samples": [],
        }

        sample_pages = min(20, len(doc))

        for page_num in range(sample_pages):
            page = doc[page_num]
            text = page.get_text()
            text_len = len(text.strip())

            page_info: Dict[str, Any] = {
                "page": page_num + 1,
                "text_length": text_len,
                "has_images": len(page.get_images()) > 0,
                "has_vector_graphics": len(page.get_drawings()) > 0,
            }

            if self.config.enable_table_detection:
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        page_info["has_tables"] = True
                        page_info["table_count"] = len(tabs.tables)
                        profile["has_tables"] = True
                        profile["estimated_tables"] += len(tabs.tables)
                except Exception:
                    pass

            if text_len < 100 and page_info["has_images"]:
                page_info["likely_scanned"] = True
                profile["has_scanned_pages"] = True

            if page_info["has_images"]:
                profile["has_images"] = True

            profile["page_types"].append(page_info)

        doc.close()

        self.document_profile = profile
        self.passes.append({
            "pass": 0,
            "strategy": "analysis",
            "profile": profile,
        })

        print(f"  Document: {profile['total_pages']} pages")
        print(f"  Tables detected: {profile['estimated_tables']}")
        print(f"  Scanned pages: {'Yes' if profile['has_scanned_pages'] else 'No'}")
        print(f"  Images present: {'Yes' if profile['has_images'] else 'No'}")

        return profile

    def pass1_text_extraction(self) -> List[Dict]:
        """Pass 1: Extract text with structure preservation."""
        print("\n📄 [Pass 1] Text extraction with structure...")

        doc = fitz.open(self.config.pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            text_blocks = self._extract_text_blocks(text_dict)
            headings = self._extract_headings(text_dict)

            pages.append({
                "page": page_num + 1,
                "text_blocks": text_blocks,
                "headings": headings,
                "raw_text": page.get_text("text"),
                "extraction_method": "pymupdf_text",
            })

        doc.close()

        self.passes.append({
            "pass": 1,
            "strategy": "text_extraction",
            "pages": len(pages),
        })

        return pages

    def _extract_text_blocks(self, text_dict: Dict) -> List[Dict]:
        """Extract text blocks with position and formatting."""
        blocks = []

        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_fonts = []

                    for span in line["spans"]:
                        line_text += span["text"]
                        line_fonts.append({
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                        })

                    if line_text.strip():
                        blocks.append({
                            "text": line_text.strip(),
                            "y_pos": block.get("bbox", [0, 0, 0, 0])[1],
                            "fonts": line_fonts,
                            "is_bold": any(f["flags"] & 2**4 for f in line_fonts),
                            "is_italic": any(f["flags"] & 2**1 for f in line_fonts),
                        })

        blocks.sort(key=lambda x: x["y_pos"])
        return blocks

    def _extract_headings(self, text_dict: Dict) -> List[Dict]:
        """Identify headings based on font size and style."""
        headings = []

        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text and len(text) < 100:
                            font_size = span.get("size", 0)
                            is_bold = span.get("flags", 0) & 2**4

                            if font_size > 12 or is_bold:
                                level = 1 if font_size > 16 else 2 if font_size > 14 else 3
                                headings.append({
                                    "text": text,
                                    "level": level,
                                    "y_pos": block.get("bbox", [0, 0, 0, 0])[1],
                                    "font_size": font_size,
                                    "is_bold": is_bold,
                                })

        headings.sort(key=lambda x: x["y_pos"])
        return headings

    def _scan_all_pages_for_tables(self) -> List[int]:
        """Find every 1-based page index that has at least one PyMuPDF-detected table."""
        pages_out: List[int] = []
        doc = fitz.open(self.config.pdf_path)
        try:
            for i in range(len(doc)):
                page = doc[i]
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        pages_out.append(i + 1)
                except Exception:
                    continue
        finally:
            doc.close()
        return pages_out

    def pass_images_extraction(self) -> List[Dict[str, Any]]:
        """
        Rasterize embedded images per page to PNG under output_dir/images/.
        Returns an inventory list (paths relative to output_dir where useful).
        """
        print("\n🖼️ [Pass images] Extracting embedded images...")
        img_dir = os.path.join(self.config.output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        inventory: List[Dict[str, Any]] = []
        doc = fitz.open(self.config.pdf_path)
        min_pixels = 400  # skip tiny icons / bullets (~20x20)
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_no = page_index + 1
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.width * pix.height < min_pixels:
                            pix = None
                            continue
                        if pix.n - pix.alpha > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        fname = f"page{page_no}_img{img_index + 1}.png"
                        out_path = os.path.join(img_dir, fname)
                        pix.save(out_path)
                        inventory.append({
                            "page": page_no,
                            "image_index": img_index + 1,
                            "xref": xref,
                            "width": pix.width,
                            "height": pix.height,
                            "file": out_path,
                        })
                        pix = None
                    except Exception as e:
                        print(f"  Warning: image xref {xref} page {page_no}: {e}")
        finally:
            doc.close()

        self.passes.append({
            "pass": "images",
            "strategy": "embedded_raster",
            "images_saved": len(inventory),
        })
        print(f"  Saved {len(inventory)} image(s) to {img_dir}")
        return inventory

    def pass2_table_extraction(self, pages_with_tables: List[int]) -> List[Dict]:
        """Pass 2: Specialized table extraction."""
        print(f"\n📊 [Pass 2] Table extraction on {len(pages_with_tables)} pages...")

        tables = []
        doc = fitz.open(self.config.pdf_path)

        for page_num in pages_with_tables:
            page = doc[page_num - 1]

            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for i, table in enumerate(tabs.tables):
                        df = table.to_pandas().fillna("")

                        table_file = os.path.join(
                            self.config.output_dir,
                            "tables",
                            f"page{page_num}_table{i+1}.csv",
                        )
                        df.to_csv(table_file, index=False)

                        tables.append({
                            "page": page_num,
                            "table_id": i,
                            "method": "pymupdf",
                            "data": df.to_dict(orient="records"),
                            "headers": df.columns.tolist(),
                            "markdown": _dataframe_to_markdown(df),
                            "num_rows": len(df),
                            "num_cols": len(df.columns),
                            "file": table_file,
                            "confidence": 0.9,
                        })
            except Exception as e:
                print(f"  Warning: Table extraction on page {page_num} failed: {e}")

            if CAMELOT_AVAILABLE and len(tables) == 0:
                try:
                    temp_pdf = os.path.join(self.config.cache_dir, f"temp_page_{page_num}.pdf")
                    with open(temp_pdf, "wb") as f:
                        f.write(page.get_pdf_output())

                    lattice_tables = camelot.read_pdf(temp_pdf, pages="1", flavor="lattice")
                    for table in lattice_tables:
                        df = table.df
                        tables.append({
                            "page": page_num,
                            "method": "camelot_lattice",
                            "data": df.to_dict(orient="records"),
                            "headers": df.iloc[0].tolist(),
                            "markdown": _dataframe_to_markdown(df),
                            "confidence": 0.95,
                        })

                    stream_tables = camelot.read_pdf(temp_pdf, pages="1", flavor="stream")
                    for table in stream_tables:
                        df = table.df
                        tables.append({
                            "page": page_num,
                            "method": "camelot_stream",
                            "data": df.to_dict(orient="records"),
                            "headers": df.iloc[0].tolist(),
                            "markdown": _dataframe_to_markdown(df),
                            "confidence": 0.85,
                        })

                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf)
                except Exception:
                    pass

        doc.close()

        self.passes.append({
            "pass": 2,
            "strategy": "table_extraction",
            "tables_found": len(tables),
        })

        return tables

    def pass3_ocr_extraction(self, scanned_pages: List[int]) -> List[Dict]:
        """Pass 3: OCR for scanned/image-heavy pages."""
        print(f"\n🖼️ [Pass 3] OCR extraction on {len(scanned_pages)} scanned pages...")

        ocr_results = []
        for page_num in scanned_pages:
            ocr_results.append({
                "page": page_num,
                "method": "ocr_needed",
                "status": "requires_manual_review",
                "note": "Page appears scanned - manual verification recommended",
            })

        self.passes.append({
            "pass": 3,
            "strategy": "ocr",
            "pages_processed": len(scanned_pages),
        })

        return ocr_results

    def pass4_cross_validation(self, extracted_data: Dict) -> Dict:
        """Pass 4: Cross-validate with different extraction tool."""
        print("\n🔄 [Pass 4] Cross-validation extraction...")

        if not PDFPLUMBER_AVAILABLE:
            print("  PDFPlumber not available, skipping cross-validation")
            return {"method": "skipped", "reason": "pdfplumber_not_available"}

        validation_results: Dict[str, Any] = {
            "method": "pdfplumber",
            "page_matches": [],
            "consistency_score": 0.0,
        }

        try:
            import pdfplumber

            with pdfplumber.open(self.config.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""

                    pass1_text = ""
                    for p in extracted_data.get("pass1", []):
                        if p.get("page") == page_num:
                            pass1_text = p.get("raw_text", "")
                            break

                    if pass1_text and text:
                        similarity = fuzz.ratio(pass1_text, text) / 100.0
                        validation_results["page_matches"].append({
                            "page": page_num,
                            "similarity": similarity,
                        })

            if validation_results["page_matches"]:
                validation_results["consistency_score"] = float(np.mean([
                    m["similarity"] for m in validation_results["page_matches"]
                ]))

        except Exception as e:
            print(f"  Cross-validation failed: {e}")

        self.passes.append({
            "pass": 4,
            "strategy": "cross_validation",
            "results": validation_results,
        })

        return validation_results

    def extract_all(self) -> Dict:
        """Orchestrate multi-pass extraction."""
        print("=" * 70)
        print("MULTI-PASS EXTRACTION ENGINE")
        print("=" * 70)

        cache_key = self._get_cache_key()
        cache_file = os.path.join(self.config.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            print("\n📦 Loading cached extraction results...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print(
            f"  Dependencies: tabulate={'yes' if TABULATE_AVAILABLE else 'NO (install for table markdown)'}, "
            f"pdfplumber={'yes' if PDFPLUMBER_AVAILABLE else 'NO'}"
        )

        profile = self.analyze_document()
        pages = self.pass1_text_extraction()

        scanned_pages = []
        for page_info in profile["page_types"]:
            if page_info.get("likely_scanned", False):
                scanned_pages.append(page_info["page"])

        if self.config.enable_table_detection and self.config.full_document_table_scan:
            print("\n📋 Scanning all pages for tables (full-document scan)...")
            pages_with_tables = self._scan_all_pages_for_tables()
            print(f"  Pages with tables: {len(pages_with_tables)}")
        else:
            pages_with_tables = []
            for page_info in profile["page_types"]:
                if page_info.get("has_tables", False):
                    pages_with_tables.append(page_info["page"])

        tables = []
        if pages_with_tables and self.config.enable_table_detection:
            tables = self.pass2_table_extraction(pages_with_tables)

        ocr_data = []
        if scanned_pages and self.config.enable_ocr:
            ocr_data = self.pass3_ocr_extraction(scanned_pages)

        cross_validation = self.pass4_cross_validation({
            "pass1": pages,
            "pass2": tables,
        })

        images: List[Dict[str, Any]] = []
        if self.config.enable_image_extraction:
            images = self.pass_images_extraction()

        extraction_result = {
            "metadata": {
                "pdf_path": self.config.pdf_path,
                "extraction_date": datetime.now().isoformat(),
                "config": asdict(self.config),
                "document_profile": profile,
                "passes_completed": len(self.passes),
            },
            "pages": pages,
            "tables": tables,
            "images": images,
            "ocr_data": ocr_data,
            "cross_validation": cross_validation,
            "extraction_log": self.passes,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(extraction_result, f)

        inv_path = os.path.join(self.config.output_dir, "image_inventory.json")
        try:
            with open(inv_path, "w", encoding="utf-8") as f:
                json.dump(images, f, indent=2, default=str)
        except OSError:
            pass

        return extraction_result

    def _get_cache_key(self) -> str:
        """Generate cache key based on file and config."""
        file_stat = os.stat(self.config.pdf_path)
        file_hash = hashlib.md5(
            f"{self.config.pdf_path}_{file_stat.st_size}_{file_stat.st_mtime}".encode()
        ).hexdigest()[:8]

        config_hash = hashlib.md5(
            json.dumps(asdict(self.config), sort_keys=True).encode()
        ).hexdigest()[:8]

        return f"extraction_{file_hash}_{config_hash}"
