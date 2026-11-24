"""Run the integrated project from raw Excel to a verified report."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Keep caches writable and CPU use predictable on local machines and CI.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from etl_rfm import load_data, clean_transactions, build_rfm
from features import refine_transactions, load_cancellations, build_customer_features
from modeling import run_modeling, metrics, RFM
from reporting import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Online Retail.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/REPORT.md")
    args = parser.parse_args()
    output, data = args.output_dir.resolve(), args.data_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    print("[1/5] Read raw Excel and build common customer features", flush=True)
    raw = load_data(args.input)
    clean = clean_transactions(raw)
    purchases = refine_transactions(clean)
    cancellations = load_cancellations(raw=raw)
    features = build_customer_features(purchases, cancellations)
    if features.empty or features.CustomerID.duplicated().any():
        raise ValueError("No eligible customers or duplicated CustomerID")
    clean.to_csv(data / "online_retail_clean.csv", index=False)
    build_rfm(clean).to_csv(data / "rfm.csv", index=False)
    input_path = data / "customer_features.csv"
    features.to_csv(input_path, index=False)
    audit = {"first_transaction": str(pd.to_datetime(raw.InvoiceDate).min()),
             "last_transaction": str(pd.to_datetime(raw.InvoiceDate).max()),
             "raw_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
             "counts": {"Raw transaction rows": len(raw), "Clean purchase rows": len(clean),
                        "Duplicate purchase rows removed": int(clean.duplicated().sum()),
                        "Product purchase rows": len(purchases), "Cancelled product rows": len(cancellations),
                        "Gross RFM customers": clean.CustomerID.nunique(),
                        "Customers before positive-net filter": purchases.CustomerID.nunique(),
                        "Final customers": len(features)}}
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print("[2/5] Select GMM on the common uncapped matrix", flush=True)
    gmm_dir = output / "gmm_bic"
    subprocess.run([sys.executable, str(ROOT / "src/gmm_modeling.py"), "--input", str(input_path),
                    "--output-dir", str(gmm_dir), "--winsor-quantile", "1.0"], check=True)
    subprocess.run([sys.executable, str(ROOT / "src/gmm_robustness.py"), "--input", str(input_path),
                    "--output-dir", str(gmm_dir), "--winsor-quantile", "1.0"], check=True)
    print("[3/5] Compare models, seeds, profiles and charts", flush=True)
    with threadpool_limits(limits=1):
        run_modeling(input_path, output, gmm_dir)
        logged = np.log1p(features[RFM])
        x = StandardScaler().fit_transform(logged)
        capped = StandardScaler().fit_transform(np.log1p(features[RFM].clip(upper=features[RFM].quantile(.99), axis=1)))
        base = KMeans(n_clusters=4, n_init=20, random_state=42).fit_predict(x)
        cap_labels = KMeans(n_clusters=4, n_init=20, random_state=42).fit_predict(capped)
        sensitivity = pd.DataFrame([
            {"Preprocessing": "No cap", "ARI_vs_uncapped": 1., **metrics(x, base)},
            {"Preprocessing": "99% upper cap", "ARI_vs_uncapped": adjusted_rand_score(base, cap_labels),
             **metrics(capped, cap_labels)},
        ])
        sensitivity.to_csv(output / "tables/preprocessing_sensitivity.csv", index=False)
    print("[4/5] Generate report from saved tables", flush=True)
    write_report(output, audit, args.report)
    print("[5/5] Verify persisted assignments and model outputs", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src/verify_project.py"), "--input", str(input_path),
                    "--output-dir", str(output)], check=True)
    print(f"DONE: {len(features):,} customers. Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
