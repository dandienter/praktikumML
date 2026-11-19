# -*- coding: utf-8 -*-
"""Render Modul_BabN_Implementasi_Python.pdf dari notebook yang sudah
dieksekusi, memakai format visual yang sama dengan Modul Bab 5
(tools/modul_pdf/build_modul.py): cover + kode + output asli + penjelasan.

Isi modul: Ringkasan Konsep, Implementasi Python, Latihan Praktikum,
Percobaan Mandiri, dan Kesimpulan (Sel 0 dan header identitas dilewati
karena sudah terwakili cover).

Pakai: python render_modul.py <nomor_bab> <judul_bab>
"""
import base64
import json
import os
import re
import sys

sys.path.insert(0, "/home/hatch/workspace/praktikumML/tools/modul_pdf")
from build_modul import CSS, cover, codeblock, outblock, fig  # noqa: E402
import markdown as mdlib  # noqa: E402
from weasyprint import HTML  # noqa: E402

REPO = "/home/hatch/workspace/praktikumML"
IMGROOT = os.path.join(REPO, "tools", "modul_pdf")


def src(c):
    return "".join(c.get("source", []))


def md_to_html(text):
    html = mdlib.markdown(text, extensions=["tables", "fenced_code"])
    # h2 seksi besar -> h2.sec; h2 praktikum/latihan/percobaan -> h3.prak
    def h2sub(m):
        inner = m.group(1)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        if re.match(r"(Praktikum|Latihan|Percobaan)", plain):
            return f'<h3 class="prak">{inner}</h3>'
        return f'<h2 class="sec">{inner}</h2>'
    html = re.sub(r"<h2>(.*?)</h2>", h2sub, html, flags=re.S)
    html = re.sub(r"<h3>(.*?)</h3>", r'<h3 class="prak">\1</h3>', html,
                  flags=re.S)
    html = re.sub(r"<h4>(.*?)</h4>", r'<h3 class="prak">\1</h3>', html,
                  flags=re.S)
    return html


def render(bab, judul):
    import glob
    folder = glob.glob(os.path.join(REPO, f"bab-{bab:02d}-*"))[0]
    folder_name = os.path.basename(folder)
    nb_path = os.path.join(folder, f"praktikum-bab-{bab:02d}.ipynb")
    nb = json.load(open(nb_path))
    imgdir = os.path.join(IMGROOT, f"img{bab}")
    os.makedirs(imgdir, exist_ok=True)
    colab = ("https://colab.research.google.com/github/dandienter/"
             f"praktikumML/blob/main/{folder_name}/"
             f"praktikum-bab-{bab:02d}.ipynb")

    body = [cover(f"Machine Learning \u2022 Bab {bab}",
                  f"Implementasi Python Bab {bab}",
                  judul,
                  ("Modul lengkap: implementasi Python sesuai buku modul, "
                   "latihan praktikum, dan percobaan mandiri, dengan "
                   "penjelasan tiap langkah, output asli hasil eksekusi, "
                   "dan interpretasi."),
                  colab)]

    mulai = False
    img_count = 0
    judul_terakhir = f"Bab {bab}"
    for c in nb["cells"]:
        t = src(c)
        if c["cell_type"] == "markdown":
            if not mulai:
                if "Ringkasan Konsep" in t or "Implementasi Python (Modul Bab" in t:
                    mulai = True
                else:
                    continue
            heads = re.findall(r"^#{2,4}\s+(.*)$", t, re.M)
            if heads:
                judul_terakhir = re.sub(r"\*\*|\(Modul Bab.*?\)", "",
                                        heads[-1]).strip()
            body.append(md_to_html(t))
        elif c["cell_type"] == "code":
            if not mulai:
                continue
            if "Sel 0" in t or "_REPO" in t:
                continue
            body.append(codeblock(t.rstrip()))
            for o in c.get("outputs", []):
                if o.get("output_type") in ("execute_result", "display_data"):
                    data = o.get("data", {})
                    if "image/png" in data:
                        img_count += 1
                        p = os.path.join(imgdir, f"fig{bab}_{img_count}.png")
                        with open(p, "wb") as f:
                            f.write(base64.b64decode(data["image/png"]))
                        body.append(fig(
                            p, f"Gambar: {judul_terakhir} (output asli "
                               "hasil eksekusi notebook)."))
                    elif "text/plain" in data:
                        txt = "".join(data["text/plain"]).strip()
                        if txt and not txt.startswith("<"):
                            body.append(outblock(txt[:2500]))
                elif o.get("output_type") == "stream":
                    txt = "".join(o.get("text", [])).strip()
                    if txt:
                        body.append(outblock(txt[:2500]))
    body.append('<p class="end">- Selesai -</p>')
    html_doc = (f'<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">'
                f"<style>{CSS}</style></head><body>{''.join(body)}</body>"
                "</html>")
    if "\u2014" in html_doc:
        raise SystemExit("EM-DASH FOUND!")
    out = os.path.join(folder, f"Modul_Bab{bab}_Implementasi_Python.pdf")
    HTML(string=html_doc).write_pdf(out)
    print("OK:", out, f"({img_count} gambar)")


if __name__ == "__main__":
    render(int(sys.argv[1]), sys.argv[2])
