"""Check saved integrated results against input, estimators and metrics."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gmm_modeling import load_rfm
from modeling import metrics, rfm_segments
from reproducibility import matches_input

ROOT = Path(__file__).resolve().parents[1]


def verify(input_path: Path, output: Path):
    rfm = load_rfm(input_path)
    summary = json.loads((output / "run_summary.json").read_text())
    if not matches_input(input_path, summary):
        raise ValueError("Input content differs from the modeling run")
    robustness_path = output / "gmm_bic/robustness_summary.json"
    robustness = json.loads(robustness_path.read_text())
    if not matches_input(input_path, robustness) or robustness["winsor_quantile"] != 1.0:
        raise ValueError("GMM robustness results use different input or preprocessing")
    saved = pd.read_csv(output / "tables/customer_segments.csv")
    input_features = pd.read_csv(input_path).sort_values("CustomerID").reset_index(drop=True)
    pd.testing.assert_frame_equal(input_features, saved[input_features.columns], check_dtype=False)
    np.testing.assert_array_equal(rfm.CustomerID, saved.CustomerID)
    np.testing.assert_allclose(rfm[["Recency", "Frequency", "Monetary"]], saved[["Recency", "Frequency", "Monetary"]])
    if len(saved) != summary["customer_count"] or not saved.CustomerID.is_unique:
        raise ValueError("Customer coverage mismatch")
    bundle = joblib.load(output / "model/kmeans.joblib")
    if not matches_input(input_path, bundle):
        raise ValueError("K-Means bundle input mismatch")
    x = bundle["scaler"].transform(np.log1p(rfm[bundle["features"]]))
    raw_labels = bundle["model"].predict(x)
    labels = np.array([bundle["raw_cluster_to_segment_id"][int(v)] for v in raw_labels])
    np.testing.assert_array_equal(labels, saved.KMeans_Cluster)
    np.testing.assert_array_equal([bundle["segment_names"][int(v)] for v in labels], saved.Segment)
    gm = joblib.load(output / "model/gmm_k4.joblib")
    probs = gm["model"].predict_proba(x)
    np.testing.assert_allclose(probs.sum(axis=1), 1.)
    np.testing.assert_allclose(probs.max(axis=1), saved.GMM_Confidence)
    np.testing.assert_array_equal([gm["raw_cluster_to_segment_id"][int(v)] for v in probs.argmax(axis=1)], saved.GMM_Cluster)
    for raw, mapped in gm["raw_cluster_to_segment_id"].items():
        np.testing.assert_allclose(probs[:, raw], saved[f"GMM_Probability_{mapped}"])
    recomputed = rfm_segments(rfm)
    for col in ["R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Segment"]:
        np.testing.assert_array_equal(recomputed[col], saved[col])
    comparison = pd.read_csv(output / "tables/model_comparison.csv").set_index("Model")
    for name, labels in [("K-Means k=4", saved.KMeans_Cluster), ("GMM k=4", saved.GMM_Cluster),
                         ("Ward k=4", saved.HC_Cluster), ("GMM BIC", saved.GMM_BIC_Cluster),
                         ("RFM scoring", saved.RFM_Segment.astype("category").cat.codes)]:
        for key, value in metrics(x, labels).items():
            np.testing.assert_allclose(value, comparison.loc[name, key], rtol=1e-8, atol=1e-10)
    profiles = pd.read_csv(output / "tables/segment_profiles.csv")
    actual = saved.groupby("KMeans_Cluster").agg(
        CustomerCount=("CustomerID", "size"), TotalMonetary=("Monetary", "sum"),
        RecencyMedian=("Recency", "median"), FrequencyMedian=("Frequency", "median"),
        MonetaryMedian=("Monetary", "median"), TenureMedian=("Tenure", "median"),
        AvgOrderValueMedian=("AvgOrderValue", "median"), ReturnRateMean=("ReturnRate", "mean"),
    )
    aligned = profiles.set_index("KMeans_Cluster").loc[actual.index]
    np.testing.assert_allclose(actual, aligned[actual.columns])
    for name in ("CustomerShare", "MonetaryShare"):
        np.testing.assert_allclose(profiles[name].sum(), 1.)
    strategies = pd.read_csv(output / "tables/segment_strategies.csv")
    if set(strategies.Segment) != set(saved.Segment):
        raise ValueError("Strategy table does not cover every segment")
    for name in ("rfm_distributions", "kmeans_selection", "ward_dendrogram", "kmeans_pca",
                 "segment_profiles", "customer_value_shares", "rfm_vs_kmeans", "model_comparison"):
        if (output / "figures" / f"{name}.png").stat().st_size == 0:
            raise ValueError(f"Empty diagnostic figure: {name}")
    print(f"PASS integrated project: {len(saved):,} customers, models, scores, metrics and profiles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/customer_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.input, args.output_dir)
    # Reuse the independent GMM checks, including posterior columns and input provenance.
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT / "src/verify_gmm.py"), "--input", str(args.input),
                    "--output-dir", str(args.output_dir / "gmm_bic")], check=True)


if __name__ == "__main__":
    main()
