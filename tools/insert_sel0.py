import json

specs = {
    "bab-01-konsep-dasar-machine-learning": ["data/iris.csv"],
    "bab-02-workflow-crisp-dm-bias-variance": ["data/breast_cancer.csv", "data/telco-customer-churn.csv"],
    "bab-03-data-preparation": ["data/telco-customer-churn.csv"],
    "bab-04-feature-engineering-pipeline": ["data/california_housing.csv", "data/titanic.csv", "data/breast_cancer.csv"],
    "bab-05-decision-tree-pruning": ["data/breast_cancer.csv"],
}
nbnums = {
    "bab-01-konsep-dasar-machine-learning": "01",
    "bab-02-workflow-crisp-dm-bias-variance": "02",
    "bab-03-data-preparation": "03",
    "bab-04-feature-engineering-pipeline": "04",
    "bab-05-decision-tree-pruning": "05",
}

for babdir, datafiles in specs.items():
    n = nbnums[babdir]
    path = f"{babdir}/praktikum-bab-{n}.ipynb"
    nb = json.load(open(path))
    idx = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")

    md = ("### Sel 0 - Persiapan Awal (Khusus Google Colab)\n\n"
          "Kalau notebook ini dibuka di Google Colab lewat link GitHub, jalankan sel di bawah ini dulu "
          "(atau langsung **Runtime > Run all**). Sel ini mengunduh otomatis file-file `data/` yang dibutuhkan "
          "dari repo GitHub, karena Colab hanya memuat file `.ipynb`-nya saja tanpa folder `data/`. "
          "Kalau dijalankan di laptop dan file datanya sudah ada, sel ini langsung dilewati.")

    code_lines = [
        "# Sel 0 - Persiapan awal (jalan otomatis paling awal saat Run All).",
        "# Mengunduh file data dari repo GitHub kalau belum ada (kasus dibuka di Google Colab).",
        "# Di laptop (folder data/ sudah ada), bagian unduh langsung dilewati.",
        "import os",
        "import urllib.request",
        "",
        f'_BAB = "{babdir}"  # nama folder bab ini di repo GitHub',
        '_REPO = "https://raw.githubusercontent.com/dandienter/praktikumML/main"',
        f"_DATA = {datafiles}  # file data yang dibutuhkan bab ini",
        "",
        "for _f in _DATA:",
        "    if os.path.exists(_f):",
        '        print(f"OK: {_f} sudah ada.")',
        "        continue",
        "    os.makedirs(os.path.dirname(_f), exist_ok=True)",
        '    print(f"mengunduh {_f} dari GitHub ...")',
        '    urllib.request.urlretrieve(f"{_REPO}/{_BAB}/{_f}", _f)',
        'print("Data siap, silakan lanjutkan.")',
    ]
    if n == "03":
        code_lines += [
            "",
            "# imbalanced-learn (SMOTE) belum tentu tersedia di Colab, pasang kalau belum ada",
            "try:",
            "    import imblearn",
            '    print("OK: imbalanced-learn sudah tersedia.")',
            "except ImportError:",
            '    print("memasang imbalanced-learn ...")',
            "    %pip install -q imbalanced-learn",
        ]
    code_src = "\n".join(code_lines)

    md_cell = {"cell_type": "markdown", "metadata": {}, "source": md}
    code_cell = {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "source": code_src, "outputs": []}
    nb["cells"][idx:idx] = [md_cell, code_cell]
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"{path}: Sel 0 disisipkan sebelum cell ke-{idx}")
