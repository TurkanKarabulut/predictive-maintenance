from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i"
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "ai4i2020.csv"

    if not csv_path.exists():
        print("Veri seti UCI kaynagindan indiriliyor...")
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            archive_bytes = response.read()

        with ZipFile(BytesIO(archive_bytes)) as archive:
            csv_name = next(
                name for name in archive.namelist()
                if Path(name).name == "ai4i2020.csv"
            )
            csv_path.write_bytes(archive.read(csv_name))

    df = pd.read_csv(csv_path)

    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Eksik sutunlar: {missing_columns}")

    if df[required].isna().any().any():
        raise ValueError("Model girdilerinde veya hedefte eksik deger var.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Hedef degisken 0 ve 1 siniflarini icermeli.")

    class_counts = {
        str(label): int(count)
        for label, count in df[TARGET].value_counts().sort_index().items()
    }

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": class_counts,
        "failure_rate": float(df[TARGET].mean()),
        "features": FEATURES,
        "target": TARGET,
        "excluded_columns": [
            column for column in df.columns
            if column not in required
        ],
        "sha256": sha256(csv_path.read_bytes()).hexdigest(),
    }

    (REPORT_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    df[FEATURES + [TARGET]].describe(include="all").to_csv(
        REPORT_DIR / "descriptive_statistics.csv",
        encoding="utf-8-sig",
    )

    source_info = {
        "name": "AI4I 2020 Predictive Maintenance Dataset",
        "source": SOURCE_PAGE,
        "download_url": DOWNLOAD_URL,
        "doi": "10.24432/C5HS5C",
        "license": "CC BY 4.0",
        "synthetic": True,
        "raw_data_modified": False,
    }
    (DATA_DIR / "source.json").write_text(
        json.dumps(source_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nVeri kontrolleri tamamlandi. Ciktilar reports klasorunde.")


if __name__ == "__main__":
    main()