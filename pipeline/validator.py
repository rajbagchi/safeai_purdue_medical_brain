"""
Extraction validation: structure, tables, cross-consistency, medical content.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from dataclasses import asdict

from .config import (
    ExtractionConfig,
    GENERAL_CLINICAL_CRITICAL_TERMS,
    ValidationReport,
)


class ExtractionValidator:
    """
    Comprehensive validation of extracted content.
    Implements multi-stage validation from CDSS architecture.
    """

    def __init__(self, extraction_result: Dict, config: ExtractionConfig):
        self.result = extraction_result
        self.config = config
        self.reports: List[ValidationReport] = []

    def validate_all(self) -> Dict:
        """Run all validation stages."""
        print("\n🔍 Running comprehensive validation...")

        validation_results: Dict[str, Any] = {}

        validation_results["structure"] = self._validate_structure()
        print(
            f"  Stage 1 (Structure): {'✅' if validation_results['structure'].passed else '❌'} "
            f"conf={validation_results['structure'].confidence:.0%}"
        )

        validation_results["tables"] = self._validate_tables()
        print(
            f"  Stage 2 (Tables): {'✅' if validation_results['tables'].passed else '❌'} "
            f"conf={validation_results['tables'].confidence:.0%}"
        )

        validation_results["cross"] = self._validate_cross_consistency()
        print(
            f"  Stage 3 (Cross): {'✅' if validation_results['cross'].passed else '❌'} "
            f"conf={validation_results['cross'].confidence:.0%}"
        )

        validation_results["medical"] = self._validate_medical_content()
        print(
            f"  Stage 4 (Medical): {'✅' if validation_results['medical'].passed else '❌'} "
            f"conf={validation_results['medical'].confidence:.0%}"
        )

        validation_results["human_review"] = self._flag_for_human_review()
        print(
            f"  Stage 5 (Human Review): flagged {len(validation_results['human_review'].issues)} items"
        )

        confidences = [
            r.confidence
            for r in validation_results.values()
            if hasattr(r, "confidence")
        ]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0

        validation_results["overall"] = {
            "confidence": overall_confidence,
            "passed": overall_confidence >= self.config.confidence_threshold,
            "needs_human_review": len(validation_results["human_review"].issues) > 0,
        }

        report_file = os.path.join(
            self.config.output_dir,
            "validation",
            f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(
                {
                    k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                    for k, v in validation_results.items()
                },
                f,
                indent=2,
                default=str,
            )

        return validation_results

    def _validate_structure(self) -> ValidationReport:
        """Validate document structure preservation."""
        issues: List[str] = []
        pages = self.result.get("pages", [])

        if not pages:
            issues.append("No pages extracted")
            return ValidationReport(
                stage="structure",
                passed=False,
                issues=issues,
                confidence=0.0,
                suggestions=["Check PDF accessibility"],
                metadata={},
            )

        page_numbers = [p["page"] for p in pages]
        expected_pages = list(range(1, len(page_numbers) + 1))
        if page_numbers != expected_pages:
            issues.append(
                f"Page numbering inconsistent: {page_numbers[:5]}..."
            )

        heading_levels = []
        for page in pages[:10]:
            for h in page.get("headings", []):
                heading_levels.append(h.get("level", 0))

        if heading_levels:
            for i in range(len(heading_levels) - 1):
                if heading_levels[i + 1] > heading_levels[i] + 1:
                    issues.append(
                        f"Heading level skipped: {heading_levels[i]} → {heading_levels[i+1]}"
                    )

        confidence = 0.9 if not issues else 0.7

        return ValidationReport(
            stage="structure",
            passed=len(issues) == 0,
            issues=issues,
            confidence=confidence,
            suggestions=(
                ["Verify heading hierarchy manually"] if issues else []
            ),
            metadata={"pages_extracted": len(pages)},
        )

    def _validate_tables(self) -> ValidationReport:
        """Validate table extraction quality."""
        issues: List[str] = []
        tables = self.result.get("tables", [])

        if not tables:
            profile = self.result.get("metadata", {}).get("document_profile", {})
            if profile.get("estimated_tables", 0) > 0:
                issues.append(
                    f"Document estimated {profile['estimated_tables']} tables but none extracted"
                )
                confidence = 0.3
            else:
                confidence = 1.0
        else:
            valid_tables = 0
            for table in tables:
                if table.get("num_cols", 0) < 2:
                    issues.append(
                        f"Table on page {table['page']} has only {table.get('num_cols', 0)} columns"
                    )
                else:
                    valid_tables += 1

                data = str(table.get("data", ""))
                if "dose" in data.lower() or "mg" in data.lower():
                    numbers = re.findall(r"\d+\.?\d*", data)
                    if len(numbers) < 3:
                        issues.append(
                            f"Possible dosing table on page {table['page']} has few numeric values"
                        )

            confidence = valid_tables / len(tables) if tables else 0.0

        return ValidationReport(
            stage="tables",
            passed=confidence >= 0.8,
            issues=issues,
            confidence=confidence,
            suggestions=(
                ["Review tables with low confidence"] if confidence < 0.8 else []
            ),
            metadata={
                "tables_extracted": len(tables),
                "valid_tables": valid_tables if tables else 0,
            },
        )

    def _validate_cross_consistency(self) -> ValidationReport:
        """Validate consistency across extraction passes."""
        issues: List[str] = []
        cross_val = self.result.get("cross_validation", {})
        consistency = cross_val.get("consistency_score", 1.0)

        if consistency < 0.9:
            issues.append(
                f"Low cross-validation consistency: {consistency:.1%}"
            )

        return ValidationReport(
            stage="cross_consistency",
            passed=consistency >= 0.9,
            issues=issues,
            confidence=consistency,
            suggestions=(
                ["Review pages with low consistency scores"]
                if consistency < 0.9
                else []
            ),
            metadata={"consistency_score": consistency},
        )

    def _validate_medical_content(self) -> ValidationReport:
        """Validate presence of critical medical content."""
        issues: List[str] = []
        all_text = ""
        for page in self.result.get("pages", []):
            for block in page.get("text_blocks", []):
                all_text += block.get("text", "") + " "

        critical_terms = (
            self.config.critical_content_terms
            if self.config.critical_content_terms
            else GENERAL_CLINICAL_CRITICAL_TERMS
        )

        found_terms: List[str] = []
        missing_terms: List[str] = []

        for term in critical_terms:
            if term.lower() in all_text.lower():
                found_terms.append(term)
            else:
                missing_terms.append(term)

        if len(missing_terms) > len(critical_terms) * 0.3:
            issues.append(
                f"Many critical medical terms missing: {missing_terms[:5]}"
            )

        confidence = 1.0 - (len(missing_terms) / len(critical_terms))

        return ValidationReport(
            stage="medical_content",
            passed=confidence >= 0.8,
            issues=issues,
            confidence=confidence,
            suggestions=(
                ["Verify medical terminology extraction"]
                if confidence < 0.8
                else []
            ),
            metadata={
                "terms_found": found_terms,
                "terms_missing": missing_terms,
            },
        )

    def _flag_for_human_review(self) -> ValidationReport:
        """Flag sections needing human verification."""
        priority_items: List[Dict[str, Any]] = []

        for table in self.result.get("tables", []):
            table_text = json.dumps(table.get("data", "")).lower()
            if any(
                word in table_text
                for word in ["dose", "mg", "tablet", "administration"]
            ):
                priority_items.append({
                    "type": "dosing_table",
                    "page": table.get("page"),
                    "reason": "Dosing accuracy critical - requires human verification",
                })

        for page in self.result.get("pages", []):
            for block in page.get("text_blocks", []):
                text = block.get("text", "").lower()
                if "contraindication" in text or "not recommended" in text:
                    priority_items.append({
                        "type": "contraindication",
                        "page": page.get("page"),
                        "reason": "Safety-critical information - verify accuracy",
                    })
                    break

        for ocr_item in self.result.get("ocr_data", []):
            if ocr_item.get("status") == "requires_manual_review":
                priority_items.append({
                    "type": "scanned_page",
                    "page": ocr_item.get("page"),
                    "reason": "OCR required - manual transcription needed",
                })

        hr_conf = max(0.0, min(1.0, 1.0 - (len(priority_items) * 0.1)))
        return ValidationReport(
            stage="human_review",
            passed=len(priority_items) == 0,
            issues=[item["reason"] for item in priority_items],
            confidence=hr_conf,
            suggestions=["Prioritize flagged items for manual review"],
            metadata={"items_for_review": priority_items},
        )
