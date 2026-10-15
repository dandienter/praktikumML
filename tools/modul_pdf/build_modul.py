#!/usr/bin/env python3
"""Rebuild modul PDF Bab 5 & 6 meniru format referensi (WeasyPrint)."""
import html as _html
from weasyprint import HTML

CSS = """
@page {
  size: A4;
  margin: 2.2cm 1.35cm 2cm 1.35cm;
  @bottom-right {
    content: counter(page);
    font-family: "DejaVu Serif", serif;
    font-size: 9pt;
    color: #555;
  }
}
@page :first {
  @bottom-right { content: none; }
  @bottom-left {
    content: "Teknik Informatika \\2022 Universitas Indraprasta PGRI";
    font-family: "DejaVu Serif", serif;
    font-size: 9pt;
    color: #333;
  }
}
html { font-family: "DejaVu Serif", serif; }
body { font-size: 10.8pt; line-height: 1.55; color: #111; text-align: justify; margin: 0; }

/* ---------- cover ---------- */
.cover { text-align: center; page-break-after: always; }
.cover .kicker {
  font-variant: small-caps; letter-spacing: 0.35em;
  font-size: 10.5pt; color: #444; margin: 6cm 0 0;
}
.cover h1 {
  font-variant: small-caps; font-weight: bold;
  font-size: 23pt; line-height: 1.4; margin: 0.7em 0 0;
}
.cover .rule {
  width: 120px; border-top: 3px solid #000; margin: 1.3em auto;
}
.cover .sub { font-style: italic; font-size: 11pt; max-width: 36em; margin: 0 auto; }
.cover .oleh { margin: 2.4em 0 0; font-size: 11pt; }
.cover .nama { font-weight: bold; font-size: 12.5pt; margin: 0.35em 0; }
.cover .npm { font-size: 11pt; margin: 0; }
.cover .colab { margin-top: 1.4em; font-size: 10pt; text-align: center; }
.cover .colab a { color: #1a4f9c; }

/* ---------- headings ---------- */
h2.sec {
  font-variant: small-caps; font-weight: bold; font-size: 15pt;
  margin: 0 0 0.55em; text-align: left;
  bookmark-level: 1; bookmark-label: content();
  break-after: avoid;
}
h3.prak {
  font-size: 12.5pt; font-weight: bold; text-align: left;
  margin: 1.25em 0 0.4em; padding-bottom: 5px;
  border-bottom: 1px solid #555;
  bookmark-level: 2; bookmark-label: content();
  break-after: avoid;
}
h3.interp {
  font-variant: small-caps; font-weight: bold; font-size: 13pt; text-align: left;
  margin: 1.4em 0 0.5em;
  bookmark-level: 1; bookmark-label: content();
  break-after: avoid;
}

/* ---------- blocks ---------- */
pre.code {
  background: #f2f2f2;
  border-top: 1px solid #7f7f7f; border-bottom: 1px solid #7f7f7f;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.6pt; line-height: 1.5;
  padding: 9px 12px; white-space: pre-wrap;
  text-align: left; break-inside: avoid;
}
div.out {
  border: 1px dashed #7f7f7f;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.6pt; line-height: 1.5;
  padding: 9px 12px; margin: 0.7em 0; white-space: pre-wrap;
  text-align: left;
}
table { border-collapse: collapse; width: 100%; margin: 0.8em 0;
        font-size: 10pt; break-inside: avoid; }
th { background: #404040; color: #fff; font-weight: bold;
     padding: 6px 8px; border: 1px solid #404040; }
td { border: 1px solid #999; padding: 5px 8px; text-align: center; }
tr:nth-child(even) td { background: #f2f2f2; }
div.fig { border: 1px solid #666; padding: 6px; margin: 0.9em 0 0.2em;
          break-inside: avoid; text-align: center; }
div.fig img { width: 100%; display: block; }
p.cap { font-style: italic; font-size: 10pt; margin: 0.35em 0 1em; text-align: left; }
div.box { border: 1px solid #333; padding: 10px 16px; margin: 0.9em 0;
          break-inside: avoid; }
p.boxtitle { font-variant: small-caps; font-weight: bold; font-size: 10.5pt;
             margin: 0 0 0.5em; text-align: left; }
p.end { text-align: center; font-style: italic; margin-top: 2.5em; }
ul { margin: 0.5em 0; padding-left: 1.4em; }
li { margin: 0.32em 0; }
p { margin: 0.55em 0; }
b.lead { font-weight: bold; }
"""

def esc(t):
    return _html.escape(t)

def cover(kicker, title1, title2, subtitle, colab_url):
    return f"""
<div class="cover">
  <p class="kicker">{kicker}</p>
  <h1>{title1}<br>{title2}</h1>
  <div class="rule"></div>
  <p class="sub">{subtitle}</p>
  <p class="oleh">Disusun oleh</p>
  <p class="nama">Ahmad Dandi Subhani</p>
  <p class="npm">202343500126</p>
  <p class="colab"><a href="{colab_url}">Buka Notebook di Google Colab</a></p>
</div>
"""

def codeblock(src):
    return f'<pre class="code">{esc(src)}</pre>'

def outblock(txt):
    return f'<div class="out">{esc(txt)}</div>'

def penjel(txt):
    return f'<p><b class="lead">Penjelasan:</b> {txt}</p>'

def h2(t):
    return f'<h2 class="sec">{t}</h2>'

def h3p(num, title):
    return f'<h3 class="prak">Praktikum {num}&nbsp;{title}</h3>'

def h3lat(num, title):
    return f'<h3 class="prak">Latihan {num}&nbsp;{title}</h3>'

def h3i(t):
    return f'<h3 class="interp">{t}</h3>'

def fig(src, caption):
    return (f'<div class="fig"><img src="file://{src}"></div>'
            f'<p class="cap">{caption}</p>')

def box(title, items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return (f'<div class="box"><p class="boxtitle">{title}</p>'
            f'<ul>{lis}</ul></div>')
