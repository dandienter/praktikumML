import json

specs = {
    "bab-01-konsep-dasar-machine-learning": "01",
    "bab-02-workflow-crisp-dm-bias-variance": "02",
    "bab-03-data-preparation": "03",
    "bab-04-feature-engineering-pipeline": "04",
    "bab-05-decision-tree-pruning": "05",
}

for babdir, n in specs.items():
    path = f"{babdir}/praktikum-bab-{n}.ipynb"
    nb = json.load(open(path))
    cells = nb["cells"]
    # cari sel Sel 0 (markdown + code yang berurutan)
    sel0_idx = next(i for i, c in enumerate(cells)
                    if c["cell_type"] == "markdown" and "Sel 0 - Persiapan Awal" in "".join(c["source"]))
    sel0 = cells[sel0_idx:sel0_idx + 2]
    assert sel0[1]["cell_type"] == "code", "struktur sel 0 tidak sesuai"
    del cells[sel0_idx:sel0_idx + 2]
    # cari heading Praktikum X.1 - Persiapan Library, sisipkan sebelumnya
    target = next(i for i, c in enumerate(cells)
                  if c["cell_type"] == "markdown"
                  and "".join(c["source"]).startswith(f"### Praktikum {int(n)}.1 - Persiapan Library"))
    cells[target:target] = sel0
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"{path}: Sel 0 dipindah ke sebelum Praktikum {int(n)}.1")
