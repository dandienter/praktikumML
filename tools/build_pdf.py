#!/usr/bin/env python3
"""build_pdf.py — mengubah notebook praktikum (.ipynb) menjadi PDF bergaya modul profesional.

Contoh:
    .venv/bin/python tools/build_pdf.py bab-01-konsep-dasar-machine-learning/praktikum-bab-01.ipynb \
        --bab 1 --judul "Konsep Dasar Machine Learning" --out bab-01-.../praktikum-bab-01.pdf
"""
import argparse
import base64
import datetime
import html as htmlmod
import importlib.metadata
import platform
import re
import sys

import nbformat
from bs4 import BeautifulSoup
from markdown import markdown as md_to_html
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from weasyprint import HTML

IDENTITAS = [
    ("Nama", "Ahmad Dandi Subhani"),
    ("NPM", "202343500126"),
    ("Kelas", "R7B"),
    ("Mata Kuliah", "Machine Learning"),
    ("Dosen", "Nurfidah Dwitiyanti, M.Si."),
]

BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def tanggal_id():
    t = datetime.date.today()
    return f"{t.day} {BULAN_ID[t.month]} {t.year}"


def versi_tools():
    info = [("Python", platform.python_version())]
    for dist, label in [("numpy", "numpy"), ("pandas", "pandas"),
                        ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
                        ("scikit-learn", "scikit-learn"),
                        ("imbalanced-learn", "imbalanced-learn")]:
        try:
            info.append((label, importlib.metadata.version(dist)))
        except importlib.metadata.PackageNotFoundError:
            info.append((label, "-"))
    return info


def slugify(text, counter):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "bagian"
    counter[s] = counter.get(s, 0) + 1
    return f"{s}-{counter[s]}"


def render_markdown_cell(src, toc_entries, counter):
    """Markdown -> HTML; kumpulkan heading h2/h3 untuk daftar isi."""
    h = md_to_html(src, extensions=["tables", "fenced_code", "toc"])
    soup = BeautifulSoup(h, "html.parser")
    for tag in soup.find_all(["h2", "h3"]):
        hid = slugify(tag.get_text(), counter)
        tag["id"] = hid
        toc_entries.append((tag.name, hid, tag.get_text()))
    # rapikan tabel markdown
    for tbl in soup.find_all("table"):
        tbl["class"] = (tbl.get("class", []) + ["md-table"])
    return str(soup)


PY_FMT = HtmlFormatter(style="default", cssclass="highlight", nowrap=False)


def render_code_cell(src, exec_count):
    highlighted = highlight(src, PythonLexer(), PY_FMT)
    label = f"In [{exec_count}]" if exec_count else "Kode"
    return (f'<div class="codecell"><div class="codehead">{label} — Kode Program</div>'
            f'<div class="codebody">{highlighted}</div></div>')


def render_outputs(outputs, fig_counter):
    parts = []
    for o in outputs:
        otype = o.get("output_type")
        if otype == "stream":
            text = "".join(o.get("text", ""))
            if text.strip():
                parts.append(f'<pre class="output">{htmlmod.escape(text)}</pre>')
        elif otype in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "image/png" in data:
                fig_counter[0] += 1
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                parts.append(
                    f'<figure class="plot"><img src="data:image/png;base64,{b64}" alt="Gambar {fig_counter[0]}"/>'
                    f'<figcaption>Gambar {fig_counter[0]}</figcaption></figure>')
            elif "text/html" in data:
                h = data["text/html"]
                if isinstance(h, list):
                    h = "".join(h)
                soup = BeautifulSoup(h, "html.parser")
                for tbl in soup.find_all("table"):
                    tbl["class"] = (tbl.get("class", []) + ["df-table"])
                # buang <style> bawaan pandas agar tidak merusak tema
                for st in soup.find_all("style"):
                    st.decompose()
                parts.append(f'<div class="dfwrap">{str(soup)}</div>')
            elif "text/plain" in data:
                t = data["text/plain"]
                if isinstance(t, list):
                    t = "".join(t)
                if t.strip():
                    parts.append(f'<pre class="output">{htmlmod.escape(t)}</pre>')
        elif otype == "error":
            parts.append('<pre class="output error">'
                         + htmlmod.escape("\n".join(o.get("traceback", []))) + '</pre>')
    return "\n".join(parts)


CSS = """
@page {
  size: A4; margin: 22mm 18mm 20mm 18mm;
  @bottom-left { content: "Praktikum Machine Learning — Bab {bab}"; font-family: 'DejaVu Sans'; font-size: 8pt; color: #888; }
  @bottom-right { content: counter(page); font-family: 'DejaVu Sans'; font-size: 8pt; color: #888; }
}
@page cover { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; }
.cover { page: cover; height: 277mm; display: block; text-align: center; padding-top: 62mm;
         background: #0f2a4a; color: #ffffff; }
.cover .kicker { font-size: 13pt; letter-spacing: 4pt; color: #9fc2e8; margin-bottom: 10mm; }
.cover h1 { font-size: 30pt; margin: 0 18mm 6mm 18mm; line-height: 1.25; }
.cover .bab { font-size: 17pt; color: #ffd166; margin-bottom: 12mm; }
.cover .ident { margin: 0 auto; width: 105mm; background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.35); border-radius: 3mm; padding: 6mm 8mm; text-align: left; }
.cover .ident table { width: 100%; border-collapse: collapse; font-size: 10.5pt; }
.cover .ident td { padding: 1.6mm 0; vertical-align: top; color: #fff; }
.cover .ident td:first-child { color: #9fc2e8; width: 32mm; }
.cover .tanggal { margin-top: 10mm; color: #9fc2e8; font-size: 11pt; }
.toc h2, .tools h2 { font-size: 16pt; color: #0f2a4a; border-bottom: 2px solid #0f2a4a; padding-bottom: 2mm; }
.tools-inline { font-size: 10.5pt; line-height: 1.8; color: #1a1a1a; margin-top: 3mm; }
.tools-inline b { color: #0f2a4a; }
.toc ul { list-style: none; padding-left: 0; }
.toc li.h2 { font-weight: bold; margin-top: 3mm; font-size: 11pt; }
.toc li.h3 { margin-left: 6mm; font-size: 10pt; }
.toc a { text-decoration: none; color: #1a1a1a; }
.toc a::after { content: leader('.') target-counter(attr(href), page); color: #666; }
h2 { font-size: 15pt; color: #0f2a4a; margin-top: 9mm; border-bottom: 1.5px solid #cfd8e3; padding-bottom: 1.5mm; }
h3 { font-size: 12.5pt; color: #1d4e89; margin-top: 6mm; }
/* jangan biarkan judul terdampar sendirian di akhir halaman */
h1, h2, h3 { break-after: avoid; break-inside: avoid; }
/* bungkus kode + outputnya agar tidak terpisah halaman */
.keep { break-inside: avoid; }
p { text-align: justify; }
img { max-width: 100%; height: auto; }
.codecell { margin: 4mm 0; border: 1px solid #d5dbe3; border-radius: 2mm; overflow: hidden; page-break-inside: avoid; }
.codehead { background: #0f2a4a; color: #fff; font-size: 8.5pt; padding: 1.8mm 4mm; font-family: 'DejaVu Sans Mono', monospace; }
.codebody { background: #f7f9fb; }
.codebody pre { margin: 0; padding: 3mm 4mm; font-size: 8.8pt; line-height: 1.45; }
.output { background: #fffdf5; border: 1px solid #e3d9b8; border-left: 3px solid #d4a017;
          border-radius: 1.5mm; padding: 3mm 4mm; font-family: 'DejaVu Sans Mono', monospace;
          font-size: 8.6pt; line-height: 1.45; white-space: pre-wrap; word-wrap: break-word; }
.output.error { border-left-color: #c0392b; background: #fdecea; }
figure.plot { margin: 5mm auto; text-align: center; page-break-inside: avoid; max-width: 100%; }
figure.plot img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 1.5mm; }
figure.plot figcaption { font-size: 9pt; color: #555; margin-top: 2mm; font-style: italic; }
table.df-table, table.md-table { border-collapse: collapse; margin: 3mm 0; font-size: 8pt; width: 100%;
  break-inside: avoid; page-break-inside: avoid; }
table.md-table { table-layout: fixed; }
table.df-table { table-layout: auto; }
table.df-table th, table.df-table td { white-space: nowrap; }
table.df-table th, table.df-table td, table.md-table th, table.md-table td { border: 1px solid #bbb; padding: 1.2mm 2mm; text-align: right;
  overflow-wrap: break-word; word-break: break-word; }
table.md-table td:first-child, table.md-table th:first-child { text-align: left; }
table.df-table thead th, table.md-table thead th { background: #0f2a4a; color: #fff; }
table.df-table tbody tr:nth-child(even), table.md-table tbody tr:nth-child(even) { background: #f2f5f9; }
.dfwrap { overflow-x: hidden; }
""" + PY_FMT.get_style_defs(".highlight") + """
/* OVERRIDE: pastikan kode panjang WRAP, tidak kepotong samping */
.codebody pre, .highlight pre, div.highlight pre {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-wrap: break-word !important;
}
pre.output {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-wrap: break-word !important;
  break-inside: auto !important;
  page-break-inside: auto !important;
}
"""


def build(nb_path, bab, judul, out_pdf):
    nb = nbformat.read(nb_path, as_version=4)
    toc_entries, counter, fig_counter = [], {}, [0]
    body_parts = []

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            src = cell.source.strip()
            if not src:
                continue
            # sel sel header identitas di bab-01 duplikat dengan cover -> tetap tampilkan, tidak masalah
            body_parts.append(render_markdown_cell(src, toc_entries, counter))
        elif cell.cell_type == "code":
            src = cell.source.strip()
            # lewati magic line %matplotlib inline agar tidak tampil di PDF
            src_lines = [l for l in src.split("\n") if l.strip() != "%matplotlib inline"]
            src = "\n".join(src_lines).strip()
            out_html = render_outputs(cell.get("outputs", []), fig_counter)
            if not src and not out_html:
                continue
            # kode (.codecell) tetap atomik; output boleh mengalir ke halaman berikut
            # supaya tidak ada halaman setengah kosong
            parts = []
            if src:
                parts.append(render_code_cell(src, cell.get("execution_count")))
            if out_html:
                parts.append(out_html)
            body_parts.append("\n".join(parts))

    # Daftar isi
    toc_items = []
    for level, hid, text in toc_entries:
        cls = "h2" if level == "h2" else "h3"
        toc_items.append(f'<li class="{cls}"><a href="#{hid}">{htmlmod.escape(text)}</a></li>')
    toc_html = ("<section class='toc'><h2>Daftar Isi</h2><ul>" + "\n".join(toc_items) + "</ul></section>"
                if toc_items else "")

    # Tools (ditulis menyamping/inline, bukan tabel — sesuai permintaan)
    tools_items = " &nbsp;&bull;&nbsp; ".join(
        f"<b>{htmlmod.escape(k)}</b> {htmlmod.escape(v)}" for k, v in versi_tools())
    tools_html = ("<section class='tools'><h2>Tools yang Digunakan</h2>"
                  f"<p class='tools-inline'>{tools_items}</p></section>")

    # Identitas cover
    ident_rows = "\n".join(
        f"<tr><td>{htmlmod.escape(k)}</td><td><b>{htmlmod.escape(v)}</b></td></tr>"
        for k, v in IDENTITAS)

    css = CSS.replace("{bab}", str(bab))
    doc = f"""<!DOCTYPE html><html lang="id"><head><meta charset="utf-8"><style>{css}</style></head><body>
<section class="cover">
  <div class="kicker">PRAKTIKUM MACHINE LEARNING</div>
  <h1>Bab {bab}<br/>{htmlmod.escape(judul)}</h1>
  <div class="bab">Kumpulan Praktikum Machine Learning</div>
  <div class="ident"><table>{ident_rows}</table></div>
  <div class="tanggal">{tanggal_id()}</div>
</section>
{toc_html}
{tools_html}
{''.join(body_parts)}
</body></html>"""

    HTML(string=doc).write_pdf(out_pdf)
    print(f"PDF ditulis: {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--bab", required=True)
    ap.add_argument("--judul", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.notebook, a.bab, a.judul, a.out)


if __name__ == "__main__":
    main()
