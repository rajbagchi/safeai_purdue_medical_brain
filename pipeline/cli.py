"""CLI entry point for the medical pipeline."""

import argparse
import os

from .config import (
    DEFAULT_UGANDA_CLINICAL_2023_PDF,
    DEFAULT_WHO_MALARIA_NIH_PDF,
    extraction_config_uganda_clinical_2023,
    extraction_config_who_malaria_nih,
)
from .orchestrator import MedicalQASystem


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Medical Q&A pipeline (WHO Malaria or Uganda Clinical Guidelines)",
    )
    parser.add_argument(
        "--preset",
        choices=("who-malaria", "uganda"),
        default=None,
        help="Use built-in paths and output dirs for a known source document",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Override PDF path (optional with --preset)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (optional with --preset)",
    )
    args = parser.parse_args()

    if args.preset == "who-malaria":
        cfg = extraction_config_who_malaria_nih(
            pdf_path=args.pdf or DEFAULT_WHO_MALARIA_NIH_PDF,
            output_dir=args.output_dir,
        )
    elif args.preset == "uganda":
        cfg = extraction_config_uganda_clinical_2023(
            pdf_path=args.pdf or DEFAULT_UGANDA_CLINICAL_2023_PDF,
            output_dir=args.output_dir,
        )
    elif args.pdf:
        cfg = None
        pdf_path = args.pdf
        output_dir = args.output_dir or "./medical_knowledge_base"
    else:
        # Default: WHO malaria at standard location (same as legacy script name)
        cfg = extraction_config_who_malaria_nih(output_dir=args.output_dir)
        pdf_path = None
        output_dir = None

    if cfg is not None:
        if not os.path.exists(cfg.pdf_path):
            print(f"❌ PDF not found: {cfg.pdf_path}")
            print("Use --pdf to set the correct path or place the file as expected.")
            return
        qa = MedicalQASystem(config=cfg)
    else:
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            return
        qa = MedicalQASystem(pdf_path, output_dir)
    qa.initialize()

    print("\n" + "=" * 70)
    print("INTERACTIVE Q&A MODE")
    print("Type 'quit' to exit, 'status' for system status")
    print("=" * 70)

    while True:
        print("\n" + "-" * 50)
        query = input("Your question: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "status":
            print("\n📊 System Status:")
            print(f"  • Chunks: {len(qa.chunks)}")
            if qa.validation_result and "overall" in qa.validation_result:
                overall = qa.validation_result["overall"]
                print(f"  • Validation confidence: {overall.get('confidence', 0):.1%}")
                print(f"  • Needs human review: {overall.get('needs_human_review', False)}")
            continue
        if not query:
            continue

        result = qa.answer(query)

        print("\n" + "=" * 70)
        print(result["response"])
        print("=" * 70)

        if not result["validation_passed"]:
            print("\n⚠️  This response has safety warnings - verify before use")
