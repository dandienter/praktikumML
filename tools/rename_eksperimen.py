import json, glob, re

OLD_INTRO = ("## Eksperimen (Coba-Coba)\n\nBagian ini isinya percobaan mandiri: ubah-ubah parameter "
             "terus amati apa yang terjadi sama hasilnya. Kayak orang lagi belajar aja.")
NEW_INTRO = ("## Percobaan Mandiri\n\nBagian ini berisi percobaan mandiri untuk proses belajar: "
             "saya mengubah-ubah parameter, lalu mengamati pengaruhnya terhadap hasil yang didapat.")

for f in sorted(glob.glob("bab-0*/praktikum-bab-*.ipynb")):
    nb = json.load(open(f))
    changed = 0
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        s = "".join(c["source"])
        orig = s
        if OLD_INTRO in s:
            s = s.replace(OLD_INTRO, NEW_INTRO)
        # bab-4 punya intro sendiri
        s = s.replace("## Eksperimen (Coba-Coba)\n\nTiga percobaan buat nguji pemahaman:",
                      "## Percobaan Mandiri\n\nTiga percobaan mandiri untuk menguji pemahaman:")
        s = s.replace("## Eksperimen (Coba-Coba)", "## Percobaan Mandiri")
        s = s.replace("### Eksperimen ", "### Percobaan Mandiri ")
        s = re.sub(r"\bEksperimen\b", "Percobaan mandiri", s)
        s = re.sub(r"\beksperimen\b", "percobaan mandiri", s)
        s = re.sub(r"\b[Cc]oba-coba\b", lambda m: "Percobaan mandiri" if m.group(0)[0].isupper() else "percobaan mandiri", s)
        if s != orig:
            c["source"] = s
            changed += 1
    json.dump(nb, open(f, "w"), indent=1, ensure_ascii=False)
    print(f, "->", changed, "cell diubah")

# verifikasi: tidak boleh ada sisa
print("\n=== sisa yang belum keganti ===")
for f in sorted(glob.glob("bab-0*/praktikum-bab-*.ipynb")):
    nb = json.load(open(f))
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"])
        for m in re.finditer(r"[Ee]ksperimen|coba-coba", s):
            print(f, "cell", i, "::", repr(s[max(0, m.start()-30):m.end()+30]))
print("verifikasi selesai")
