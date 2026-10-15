#!/usr/bin/env python3
"""Render Modul_Bab5_Implementasi_Python.pdf (Bab 5 saja)."""
import sys
sys.path.insert(0, "/home/hatch/workspace/praktikumML/tools/modul_pdf")
from build_modul import CSS
from weasyprint import HTML

import bab5_content

OUT = ("/home/hatch/workspace/praktikumML/bab-05-decision-tree-pruning/"
       "Modul_Bab5_Implementasi_Python.pdf")

body = bab5_content.body()
if "\u2014" in body:
    raise SystemExit("EM-DASH FOUND!")
doc = (f'<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">'
       f"<style>{CSS}</style></head><body>{body}</body></html>")
HTML(string=doc).write_pdf(OUT)
print("OK:", OUT)
