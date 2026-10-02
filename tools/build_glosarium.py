#!/usr/bin/env python3
"""Bangun GLOSARIUM.pdf dari GLOSARIUM.md dengan cover yang rapi."""
import markdown
from weasyprint import HTML

with open("GLOSARIUM.md") as f:
    md_text = f.read()

lines = md_text.split("\n")
# buang H1 dan paragraf intro (sudah terwakili di cover)
out = []
skip_intro = True
for l in lines:
    if l.startswith("# Glosarium"):
        continue
    if skip_intro:
        if l.startswith("## Tools"):
            skip_intro = False
            out.append(l)
        continue
    out.append(l)
body_html = markdown.markdown("\n".join(out))

CSS = """
@page {
  size: A4; margin: 22mm 18mm 20mm 18mm;
  @bottom-left { content: "Praktikum Machine Learning - Glosarium"; font-family: 'DejaVu Sans'; font-size: 8pt; color: #888; }
  @bottom-right { content: counter(page); font-family: 'DejaVu Sans'; font-size: 8pt; color: #888; }
}
@page :first {
  margin: 0;
  background: #1e3a5f;
  @bottom-left { content: none; }
  @bottom-right { content: none; }
}
body { font-family: 'DejaVu Sans'; font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; }
h2 { color: #1e3a5f; border-bottom: 2px solid #1e3a5f; padding-bottom: 4px; margin-top: 28px; break-after: avoid; }
ul { padding-left: 20px; }
li { margin-bottom: 6px; }
li strong { color: #1e3a5f; }
code { font-family: 'DejaVu Sans Mono'; font-size: 9pt; background: #f0f2f5; padding: 1px 4px; border-radius: 3px; }
.cover { position: relative; height: 277mm; text-align: center; overflow: hidden; page-break-after: always; }
.cover-bg-letter {
  position: absolute; top: 40px; left: 0; right: 0;
  font-size: 340pt; font-weight: bold; color: rgba(255,255,255,0.04);
  line-height: 1; pointer-events: none;
}
.cover-inner { position: relative; padding-top: 90px; }
.cover .kicker { letter-spacing: 5px; color: #8fa3bf; font-size: 10pt; margin: 0 0 18px 0; }
.cover .gold-rule { width: 64px; height: 3px; background: #f0b429; margin: 0 auto 30px auto; }
.cover h1 { color: white; font-size: 44pt; letter-spacing: 8px; margin: 0 0 14px 0; font-weight: bold; }
.cover .sub { color: #f0b429; font-size: 15pt; margin: 0 0 22px 0; line-height: 1.5; }
.cover .desc { color: #aebdd2; font-size: 10.5pt; line-height: 1.7; margin: 0 0 44px 0; }
.cover .ident-box {
  display: inline-block; text-align: left;
  border: 1px solid rgba(255,255,255,0.25); border-radius: 8px;
  padding: 18px 34px; color: #dbe4f0; font-size: 10pt; line-height: 2;
}
.cover .ident-box .lbl { color: #8fa3bf; display: inline-block; width: 78px; }
.cover .date { color: #8fa3bf; font-size: 10pt; margin-top: 36px; }
"""

cover = """
<div class="cover">
  <div class="cover-bg-letter">Aa</div>
  <div class="cover-inner">
    <p class="kicker">PRAKTIKUM MACHINE LEARNING</p>
    <div class="gold-rule"></div>
    <h1>GLOSARIUM</h1>
    <p class="sub">Istilah &amp; Tools<br>Bab 1 sampai 13</p>
    <p class="desc">Panduan cepat memahami istilah dan perangkat<br>yang dipakai di seluruh praktikum</p>
    <div class="ident-box">
      <span class="lbl">Nama</span> Ahmad Dandi Subhani<br>
      <span class="lbl">NPM</span> 202343500126<br>
      <span class="lbl">Kelas</span> R7B<br>
      <span class="lbl">Dosen</span> Nurfidah Dwitiyanti, M.Si.
    </div>
    <p class="date">2 Oktober 2026</p>
  </div>
</div>
"""

full = f"<html><head><style>{CSS}</style></head><body>{cover}{body_html}</body></html>"
HTML(string=full).write_pdf("GLOSARIUM.pdf")
print("PDF ditulis: GLOSARIUM.pdf")
