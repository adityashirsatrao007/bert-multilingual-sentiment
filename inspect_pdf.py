
from pypdf import PdfReader
import os
import re

# Find the PDF file
files = [f for f in os.listdir('.') if f.endswith('.pdf')]
if not files:
    print("No PDF file found.")
    exit()

pdf_path = os.path.abspath(files[0])
print(f"Analyzing PDF: {pdf_path}")

reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# Stats
total_chars = len(full_text)
words = full_text.split()
total_words = len(words)

print("-" * 30)
print(f"Total Pages: {len(reader.pages)}")
print(f"Total Character Count: {total_chars}")
print(f"Total Word Count: {total_words}")
print("-" * 30)

# Try to find headings (uppercase lines or specific patterns)
print("Potential Structure (Headings):")
lines = full_text.split('\n')
for i, line in enumerate(lines):
    line = line.strip()
    if not line: continue
    
    # Heuristics for headings in PDF extractions (often all caps, or numbered)
    # Check for likely section headers
    if (len(line) < 50 and line.isupper() and len(line) > 3):
        print(f"  {line}")
    elif re.match(r'^\d+\.\s+[A-Z]', line): # Matches 1. INTRODUCTION
        print(f"  {line}")
    elif re.match(r'^[IVX]+\.\s+[A-Z]', line): # Matches I. INTRODUCTION
        print(f"  {line}")

print("-" * 30)
