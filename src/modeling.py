"""Common-matrix segmentation, diagnostics, persistence and reporting."""

from itertools import combinations
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from gmm_modeling import load_rfm
from reproducibility import file_fingerprint, software_versions, matches_input

RFM = ["Recency", "Frequency", "Monetary"]
SEED = 42


def quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Equal values always get equal scores; tied groups need not be equal sized.

    Empirical quantile boundaries can repeat for discrete Frequency. Search
    with side='left' assigns a boundary value to the lower bin consistently.
    """
    boundaries = series.quantile([.2, .4, .6, .8]).to_numpy()
    scores = pd.Series(np.searchsorted(boundaries, series, side="left") + 1, index=series.index)
    return scores if ascending else 6 - scores


def rfm_segments(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    for col in RFM:
        customers[f"{col[0]}_Score"] = quintile_score(customers[col], col != "Recency")
    customers["RFM_Score"] = customers[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    rules = {
        r"[1-2][1-2]": "Hibernating", r"[1-2][3-4]": "At Risk",
        r"[1-2]5": "Can't Lose Them", r"3[1-2]": "About to Sleep",
        r"33": "Need Attention", r"[3-4][4-5]": "Loyal", r"41": "Promising",
        r"51": "New", r"[4-5][2-3]": "Potential Loyalists", r"5[4-5]": "Champions",
    }
    code = customers["R_Score"].astype(str) + customers["F_Score"].astype(str)
    customers["RFM_Segment"] = code.replace(rules, regex=True)
    if customers["RFM_Segment"].str.fullmatch(r"\d\d").any():
        raise ValueError("Unmapped RFM score")
    return customers


def relabel(labels: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
    centers = pd.DataFrame(x, columns=RFM).groupby(labels).mean()
    order = (-centers.Recency + centers.Frequency + centers.Monetary).sort_values(ascending=False).index
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    return np.array([mapping[int(label)] for label in labels]), mapping


def metrics(x: np.ndarray, labels: np.ndarray) -> dict:
    counts = np.unique(labels, return_counts=True)[1]
    if not 1 < len(counts) < len(labels):
        raise ValueError("Metrics need at least two clusters and fewer clusters than customers")
    return {
        "Components": len(counts), "Silhouette": float(silhouette_score(x, labels)),
        "DaviesBouldin": float(davies_bouldin_score(x, labels)),
        "CalinskiHarabasz": float(calinski_harabasz_score(x, labels)),
        "SmallestSegment": int(counts.min()), "LargestSegment": int(counts.max()),
        "SmallestShare": float(counts.min() / len(labels)),
        "LargestShare": float(counts.max() / len(labels)),
    }


def assign_names(customers: pd.DataFrame) -> dict:
    profile = customers.groupby("KMeans_Cluster")[[*RFM, "Tenure"]].median()
    vip = profile.Monetary.idxmax()
    dormant = profile.drop(index=vip).Recency.idxmax()
    rest = profile.drop(index=[vip, dormant]).sort_values("Tenure")
    return {int(vip): "VIP", int(dormant): "Ngu dong",
            int(rest.index[0]): "Khach moi", int(rest.index[1]): "Trung thanh tiem nang"}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figures(customers, x, profile, comparison, selection, tree, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i, col in enumerate(RFM):
        axes[0, i].hist(customers[col], bins=45)
        axes[0, i].set_title(col)
        axes[1, i].hist(np.log1p(customers[col]), bins=45)
        axes[1, i].set_title(f"log1p({col})")
    save_figure(fig, output / "rfm_distributions.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["Inertia", "Silhouette", "DaviesBouldin"]):
        ax.plot(selection.k, selection[col], "o-")
        ax.axvline(4, color="orange", linestyle="--")
        ax.set(xlabel="k", title=col)
    save_figure(fig, output / "kmeans_selection.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(tree, truncate_mode="lastp", p=25, ax=ax)
    # n-k merges yield k clusters; the next merge would yield k-1.
    cut_height = (tree[-4, 2] + tree[-3, 2]) / 2
    ax.axhline(cut_height, linestyle="--", color="orange", label="k=4")
    ax.set_title("Ward dendrogram (last 25 branches)")
    ax.legend()
    save_figure(fig, output / "ward_dendrogram.png")
    coords = PCA(n_components=2, random_state=SEED).fit_transform(x)
    fig, ax = plt.subplots(figsize=(9, 6))
    for segment, group in customers.groupby("Segment"):
        idx = group.index.to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], s=12, alpha=.5, label=segment)
    ax.set(xlabel="PC1", ylabel="PC2", title="K-Means segments on PCA projection")
    ax.legend()
    save_figure(fig, output / "kmeans_pca.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    means = pd.DataFrame(x, columns=RFM).groupby(customers.Segment).mean()
    for segment, row in means.iterrows():
        ax.plot(RFM, row, "o-", label=segment)
    ax.set_ylabel("Mean standardized log1p feature")
    ax.legend()
    save_figure(fig, output / "segment_profiles.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    profile.set_index("Segment")[["CustomerShare", "MonetaryShare"]].mul(100).plot.barh(ax=ax)
    ax.set_xlabel("Share (%)")
    save_figure(fig, output / "customer_value_shares.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pd.crosstab(customers.Segment, customers.RFM_Segment, normalize="index") * 100,
                annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("RFM baseline within K-Means segments (%)")
    save_figure(fig, output / "rfm_vs_kmeans.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    comparison.set_index("Model").Silhouette.plot.barh(ax=ax)
    ax.set_xlabel("Silhouette (all customers, same matrix)")
    save_figure(fig, output / "model_comparison.png")


def run_modeling(input_path: Path, output: Path, gmm_dir: Path) -> dict:
    gmm_summary = json.loads((gmm_dir / "run_summary.json").read_text())
    if not matches_input(input_path, gmm_summary):
        raise ValueError("GMM BIC input differs from common customer features")
    if gmm_summary["preprocessing"]["winsor_quantile"] != 1.0:
        raise ValueError("The common comparison requires uncapped GMM features")
    validated = load_rfm(input_path)
    features = pd.read_csv(input_path).sort_values("CustomerID").reset_index(drop=True)
    if not np.array_equal(features.CustomerID, validated.CustomerID):
        raise ValueError("Customer alignment failed")
    if len(features) < 30:
        raise ValueError("The full comparison needs at least 30 customers")
    for name in ("tables", "figures", "model"):
        (output / name).mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    customers = rfm_segments(features)
    customers[RFM].describe().to_csv(tables / "rfm_descriptive_statistics.csv")
    customers[RFM].corr(method="spearman").to_csv(tables / "rfm_spearman_correlation.csv")
    scaler = StandardScaler()
    x = scaler.fit_transform(np.log1p(customers[RFM]))
    candidates = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(x)
        candidates.append({"k": k, "Inertia": float(km.inertia_), **metrics(x, km.labels_)})
    selection = pd.DataFrame(candidates)
    selection.to_csv(tables / "kmeans_selection.csv", index=False)
    km = KMeans(n_clusters=4, n_init=20, random_state=SEED).fit(x)
    customers["KMeans_Cluster"], mapping = relabel(km.labels_, x)
    names = assign_names(customers)
    customers["Segment"] = customers.KMeans_Cluster.map(names)
    gmm = GaussianMixture(n_components=4, covariance_type="full", reg_covar=.1,
                          n_init=30, max_iter=1000, random_state=SEED).fit(x)
    if not gmm.converged_:
        raise RuntimeError("Fixed-k GMM did not converge")
    customers["GMM_Cluster"], gmm_mapping = relabel(gmm.predict(x), x)
    probs = gmm.predict_proba(x)
    customers["GMM_Confidence"] = probs.max(axis=1)
    for raw, label in gmm_mapping.items():
        customers[f"GMM_Probability_{label}"] = probs[:, raw]
    tree = linkage(x, method="ward")
    customers["HC_Cluster"], _ = relabel(fcluster(tree, 4, criterion="maxclust"), x)
    saved = pd.read_csv(gmm_dir / "tables/customer_segments.csv")
    bic_labels = customers[["CustomerID"]].merge(saved[["CustomerID", "SegmentID"]],
                                                 on="CustomerID", how="left", validate="one_to_one")
    if bic_labels.SegmentID.isna().any():
        raise ValueError("BIC-selected GMM does not cover the common customer set")
    customers["GMM_BIC_Cluster"] = bic_labels.SegmentID.to_numpy()
    all_labels = {
        "RFM scoring": customers.RFM_Segment.astype("category").cat.codes.to_numpy(),
        "K-Means k=4": customers.KMeans_Cluster.to_numpy(),
        "GMM k=4": customers.GMM_Cluster.to_numpy(),
        "Ward k=4": customers.HC_Cluster.to_numpy(),
        "GMM BIC": customers.GMM_BIC_Cluster.to_numpy(),
    }
    comparison = pd.DataFrame([{"Model": name, **metrics(x, labels)} for name, labels in all_labels.items()])
    comparison.to_csv(tables / "model_comparison.csv", index=False)
    agreement = pd.DataFrame([
        {"ModelA": a, "ModelB": b, "ARI": adjusted_rand_score(all_labels[a], all_labels[b])}
        for a, b in combinations(all_labels, 2)
    ])
    agreement.to_csv(tables / "model_agreement.csv", index=False)
    stability = []
    for name, estimator in [
        ("K-Means k=4", lambda seed: KMeans(n_clusters=4, n_init=20, random_state=seed)),
        ("GMM k=4", lambda seed: GaussianMixture(n_components=4, covariance_type="full", reg_covar=.1,
                                                n_init=30, max_iter=1000, random_state=seed)),
    ]:
        runs = {}
        for seed in range(5):
            fitted = estimator(seed).fit(x)
            if hasattr(fitted, "converged_") and not fitted.converged_:
                raise RuntimeError(f"{name} seed {seed} did not converge")
            runs[seed] = fitted.predict(x)
        for a, b in combinations(runs, 2):
            stability.append({"Model": name, "SeedA": a, "SeedB": b,
                              "ARI": adjusted_rand_score(runs[a], runs[b])})
    pd.DataFrame(stability).to_csv(tables / "seed_stability.csv", index=False)
    profile = customers.groupby(["KMeans_Cluster", "Segment"]).agg(
        CustomerCount=("CustomerID", "size"), RecencyMedian=("Recency", "median"),
        FrequencyMedian=("Frequency", "median"), MonetaryMedian=("Monetary", "median"),
        TenureMedian=("Tenure", "median"), AvgOrderValueMedian=("AvgOrderValue", "median"),
        ReturnRateMean=("ReturnRate", "mean"), TotalMonetary=("Monetary", "sum"),
    ).reset_index()
    profile["CustomerShare"] = profile.CustomerCount / len(customers)
    profile["MonetaryShare"] = profile.TotalMonetary / profile.TotalMonetary.sum()
    profile.to_csv(tables / "segment_profiles.csv", index=False)
    strategies = {
        "VIP": ("Giu chan, quyen loi thanh vien, gioi thieu khach", "Repeat purchase rate; retention; contribution margin"),
        "Trung thanh tiem nang": ("Nhac mua lai, cross-sell, uu dai theo nguong", "Repeat purchase rate; net spend; margin"),
        "Khach moi": ("Onboarding, thu nghiem uu dai lan mua thu hai", "Second purchase within 60 days"),
        "Ngu dong": ("Thu nghiem win-back chi phi thap", "Reactivation rate; cost per reactivated customer"),
    }
    strategy_table = profile[["Segment", "CustomerCount"]].copy()
    strategy_table["ActionHypothesis"] = strategy_table.Segment.map(lambda n: strategies[n][0])
    strategy_table["KPI"] = strategy_table.Segment.map(lambda n: strategies[n][1])
    strategy_table.to_csv(tables / "segment_strategies.csv", index=False)
    customers.to_csv(tables / "customer_segments.csv", index=False)
    bundle = {"model": km, "scaler": scaler, "features": RFM, "raw_cluster_to_segment_id": mapping,
              "segment_names": names, "software_versions": software_versions(),
              "input_sha256": file_fingerprint(input_path), "fingerprint_format": "sha256-lf-v1"}
    joblib.dump(bundle, output / "model/kmeans.joblib")
    joblib.dump({"model": gmm, "scaler": scaler, "features": RFM,
                 "raw_cluster_to_segment_id": gmm_mapping}, output / "model/gmm_k4.joblib")
    make_figures(customers, x, profile, comparison, selection, tree, output / "figures")
    summary = {"customer_count": len(customers), "features": RFM,
               "input_sha256": file_fingerprint(input_path), "fingerprint_format": "sha256-lf-v1",
               "preprocessing": "log1p then StandardScaler; no winsorization",
               "seed": SEED, "primary_model": "K-Means k=4", "software_versions": software_versions(),
               "comparison": comparison.to_dict(orient="records"),
               "gmm_k4_bic": float(gmm.bic(x)),
               "gmm_k4_share_confidence_below_0_7": float((probs.max(axis=1) < .7).mean())}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
