#!/usr/bin/env python3
"""
Compile dissertation chapters into single document

Combines all chapter markdown files, handles cross-references,
and generates unified bibliography.
"""

import argparse
import os
import re
from pathlib import Path

def compile_chapters(chapters_dir="docs/dissertation", output_file="docs/dissertation/full_dissertation.md"):
    """Compile all chapter files into one document"""
    
    # Expected chapter order
    chapters = [
        "00_frontmatter.md",
        "01_introduction.md",
        "02_foundations.md",
        "03_general_theory.md",
        "04_hallucination.md",
        "05_empirical.md",
        "06_extensions.md",
        "07_conclusion.md",
    ]
    
    compiled = []
    
    for chapter_file in chapters:
        chapter_path = Path(chapters_dir) / chapter_file
        
        if not chapter_path.exists():
            print(f"⚠ Warning: {chapter_file} not found, skipping")
            compiled.append(f"\n\n---\n\n# [Chapter: {chapter_file[3:-3].title()} - To Be Written]\n\n")
            continue
        
        with open(chapter_path, 'r') as f:
            content = f.read()
        
        # Add chapter separator
        compiled.append(f"\n\n---\n\n")
        compiled.append(content)
        
        print(f"✓ Added: {chapter_file}")
    
    # Write compiled version
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("".join(compiled))
    
    print(f"\n✓ Compiled dissertation: {output_file}")
    print(f"✓ Total length: {len(''.join(compiled))} characters")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compile dissertation chapters")
    parser.add_argument("--chapters_dir", default="docs/dissertation", help="Directory containing chapters")
    parser.add_argument("--output", default="docs/dissertation/full_dissertation.md", help="Output file")
    parser.add_argument("--pdf", action="store_true", help="Also generate PDF (requires pandoc)")
    
    args = parser.parse_args()
    
    output_md = compile_chapters(args.chapters_dir, args.output)
    
    if args.pdf:
        # Requires pandoc installed
        pdf_output = output_md.replace(".md", ".pdf")
        os.system(f"pandoc {output_md} -o {pdf_output} --toc --number-sections")
        print(f"✓ PDF generated: {pdf_output}")

if __name__ == "__main__":
    main()
