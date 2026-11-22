"""Vẽ bộ hình cho báo cáo Word và slide thuyết trình.

Chỉ ĐỌC kết quả mà `python src/pipeline.py` đã lưu, không huấn luyện lại và
không ghi đè `outputs/figures/` của pipeline. Hình mới nằm ở
`outputs/figures_report/`.

Chạy từ thư mục gốc repo (sau khi pipeline đã chạy xong):

    python src/visualize_report.py
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RFM = ["Recency", "Frequency", "Monetary"]
SEED = 42

# Tên có dấu + thứ tự + màu cố định cho từng nhóm, dùng chung cho MỌI hình.
SEGMENT_NAMES = {
    "VIP": "VIP",
    "Trung thanh tiem nang": "Trung thành tiềm năng",
    "Khach moi": "Khách mới",
    "Ngu dong": "Ngủ đông",
}
SEGMENT_ORDER = ["VIP", "Trung thành tiềm năng", "Khách mới", "Ngủ đông"]
SEGMENT_COLORS = {
    "VIP": "#C0392B",
    "Trung thành tiềm năng": "#2E86C1",
    "Khách mới": "#27AE60",
    "Ngủ đông": "#95A5A6",
}
FEATURE_NAMES = {
    "Recency": "Recency (ngày từ lần mua cuối)",
    "Frequency": "Frequency (số hóa đơn)",
    "Monetary": "Monetary (chi tiêu ròng, £)",
}
AUDIT_NAMES = {
    "Raw transaction rows": "Dữ liệu gốc",
    "Clean purchase rows": "Bỏ thiếu ID, đơn hủy, SL/giá ≤ 0",
    "Product purchase rows": "Chỉ giữ mã sản phẩm, bỏ trùng",
    "Gross RFM customers": "Khách có CustomerID",
    "Customers before positive-net filter": "Khách mua sản phẩm hợp lệ",
    "Final customers": "Khách có chi tiêu ròng > 0",
}
MODEL_NAMES = {"RFM scoring": "RFM scoring – quy tắc"}

ACCENT = "#2C3E50"
MUTED = "#7F8C8D"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",  # có đủ dấu tiếng Việt, đi kèm matplotlib
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png", flush=True)


def fmt_int(x) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def thousands(x, _pos=None) -> str:
    return fmt_int(x)


def load_inputs(root: Path):
    out = root / "outputs"
    segments = pd.read_csv(out / "tables/customer_segments.csv")
    segments["Segment"] = segments["Segment"].map(SEGMENT_NAMES).fillna(segments["Segment"])
    profiles = pd.read_csv(out / "tables/segment_profiles.csv")
    profiles["Segment"] = profiles["Segment"].map(SEGMENT_NAMES).fillna(profiles["Segment"])
    profiles = profiles.set_index("Segment").loc[SEGMENT_ORDER].reset_index()
    selection = pd.read_csv(out / "tables/kmeans_selection.csv")
    comparison = pd.read_csv(out / "tables/model_comparison.csv")
    audit = json.loads((out / "data_audit.json").read_text())
    unknown = set(segments["Segment"]) - set(SEGMENT_ORDER)
    if unknown:
        raise ValueError(f"Nhóm chưa có tên/màu trong SEGMENT_NAMES: {sorted(unknown)}")
    return segments, profiles, selection, comparison, audit


# ---------------------------------------------------------------- EDA

def fig_cleaning_funnel(audit: dict, out: Path) -> None:
    counts = audit["counts"]
    rows = ["Raw transaction rows", "Clean purchase rows", "Product purchase rows"]
    custs = ["Gross RFM customers", "Customers before positive-net filter", "Final customers"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"wspace": 0.75})
    for ax, keys, title, color in [
        (axes[0], rows, "Số dòng giao dịch", ACCENT),
        (axes[1], custs, "Số khách hàng", "#2E86C1"),
    ]:
        labels = [AUDIT_NAMES[k] for k in keys][::-1]
        values = [counts[k] for k in keys][::-1]
        bars = ax.barh(labels, values, color=color, alpha=0.85)
        top = values[-1]
        for bar, v in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f"  {fmt_int(v)} ({v / top:.0%})", va="center", fontsize=10)
        ax.set_title(title)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(4))
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(thousands))
        ax.set_xlim(0, top * 1.45)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Dữ liệu còn lại sau từng bước tiền xử lý", fontsize=15, fontweight="bold", y=1.03)
    save(fig, out, "01_eda_cleaning_funnel")


def load_transactions(root: Path) -> pd.DataFrame:
    tx = pd.read_csv(root / "data/online_retail_clean.csv",
                     usecols=["InvoiceNo", "InvoiceDate", "Country", "TotalPrice", "CustomerID"],
                     parse_dates=["InvoiceDate"])
    return tx


def fig_monthly_revenue(tx: pd.DataFrame, out: Path) -> None:
    month = tx["InvoiceDate"].dt.to_period("M")
    monthly = tx.groupby(month).agg(Revenue=("TotalPrice", "sum"), Invoices=("InvoiceNo", "nunique"))
    last_day = tx["InvoiceDate"].max()
    x = np.arange(len(monthly))
    labels = [p.strftime("%m/%Y") for p in monthly.index]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = [ACCENT] * len(monthly)
    colors[-1] = "#BDC3C7"  # tháng cuối chưa đủ dữ liệu
    ax.bar(x, monthly["Revenue"] / 1000, color=colors)
    ax.set_ylabel("Doanh thu (nghìn £)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(x, monthly["Invoices"], color="#E67E22", marker="o", lw=2, label="Số hóa đơn")
    ax2.set_ylabel("Số hóa đơn", color="#E67E22")
    ax2.tick_params(axis="y", colors="#E67E22")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, monthly["Invoices"].max() * 1.15)
    peak = int(np.argmax(monthly["Revenue"].values[:-1]))
    peak_value = monthly["Revenue"].iloc[peak]
    ax.annotate(f"Cao điểm {labels[peak]}\n≈ {peak_value / 1e6:.2f} triệu £".replace(".", ","),
                xy=(peak - 0.4, peak_value / 1000),
                xytext=(peak - 4.2, peak_value / 1000 * 0.93),
                arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=10)
    ax.text(x[-1], monthly["Revenue"].iloc[-1] / 1000, f"chỉ đến\n{last_day:%d/%m}",
            ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_title("Doanh thu và số hóa đơn theo tháng (12/2010 – 12/2011)")
    ax2.legend(loc="upper left")
    save(fig, out, "02_eda_monthly_revenue")


def fig_countries(tx: pd.DataFrame, out: Path, top: int = 10) -> None:
    by_country = tx.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
    total = by_country.sum()
    uk_share = by_country.get("United Kingdom", 0) / total
    customers = tx.groupby("Country")["CustomerID"].nunique()
    others = by_country.drop("United Kingdom", errors="ignore").head(top)
    others = others.rename(index={"EIRE": "Ireland (EIRE)"})
    customers = customers.rename(index={"EIRE": "Ireland (EIRE)"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2.2]})
    wedges, _texts, _pcts = ax1.pie(
        [uk_share, 1 - uk_share], colors=[ACCENT, "#BDC3C7"], autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(width=0.45), textprops=dict(fontsize=12, color="white", fontweight="bold"),
        pctdistance=0.78)
    ax1.legend(wedges, ["Anh (UK)", "Các nước khác"], loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax1.set_title("Tỷ trọng doanh thu")
    ax1.axis("equal")
    y = np.arange(len(others))[::-1]
    ax2.barh(y, others.values / 1000, color="#2E86C1")
    ax2.set_yticks(y, others.index)
    for yi, (country, v) in zip(y, others.items()):
        ax2.text(v / 1000, yi, f"  {v / total:.1%} · {customers[country]} khách", va="center", fontsize=9)
    ax2.set_xlabel("Doanh thu (nghìn £)")
    ax2.set_xlim(0, others.max() / 1000 * 1.35)
    ax2.set_title(f"Top {top} quốc gia ngoài Anh")
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Doanh thu tập trung gần như hoàn toàn ở thị trường Anh",
                 fontsize=15, fontweight="bold", y=1.03)
    save(fig, out, "03_eda_countries")


def fig_rfm_distributions(segments: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for j, col in enumerate(RFM):
        raw = segments[col]
        axes[0, j].hist(raw, bins=50, color=MUTED)
        axes[0, j].set_title(col)
        axes[0, j].text(0.97, 0.92, f"độ lệch = {raw.skew():.1f}", transform=axes[0, j].transAxes,
                        ha="right", fontsize=10, color=ACCENT)
        axes[0, j].xaxis.set_major_locator(mtick.MaxNLocator(5))
        axes[0, j].xaxis.set_major_formatter(mtick.FuncFormatter(thousands))
        logged = np.log1p(raw)
        axes[1, j].hist(logged, bins=40, color="#2E86C1")
        axes[1, j].set_xlabel(f"log(1 + {col})")
        axes[1, j].text(0.97, 0.92, f"độ lệch = {logged.skew():.1f}", transform=axes[1, j].transAxes,
                        ha="right", fontsize=10, color=ACCENT)
    axes[0, 0].set_ylabel("Giá trị gốc\nSố khách")
    axes[1, 0].set_ylabel("Sau log1p\nSố khách")
    fig.suptitle("Phân phối RFM trước và sau biến đổi log1p", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "04_rfm_distributions")


# ---------------------------------------------------------------- Mô hình

def fig_kmeans_selection(selection: pd.DataFrame, out: Path, chosen: int = 4) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(selection["k"], selection["Inertia"], marker="o", color=ACCENT)
    ax1.set(title="Phương pháp khuỷu tay (Elbow)", xlabel="Số cụm k", ylabel="Inertia (WCSS)")
    ax2.plot(selection["k"], selection["Silhouette"], marker="o", color="#2E86C1", label="Silhouette ↑")
    ax2.plot(selection["k"], selection["DaviesBouldin"], marker="s", color="#E67E22", label="Davies–Bouldin ↓")
    ax2.set(title="Chỉ số chất lượng cụm", xlabel="Số cụm k")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.axvline(chosen, color=SEGMENT_COLORS["VIP"], ls="--", lw=1.5)
        ax.set_xticks(selection["k"])
    ax1.text(chosen + 0.15, ax1.get_ylim()[1] * 0.95, f"chọn k = {chosen}",
             color=SEGMENT_COLORS["VIP"], fontsize=10, va="top")
    fig.suptitle("Chọn số cụm cho K-Means", fontsize=15, fontweight="bold", y=1.02)
    save(fig, out, "05_kmeans_selection")


def fig_model_comparison(comparison: pd.DataFrame, out: Path) -> None:
    data = comparison.copy()
    data["Model"] = data["Model"].replace(MODEL_NAMES)
    main = "K-Means k=4"
    colors = [SEGMENT_COLORS["VIP"] if m == main else "#AEB6BF" for m in data["Model"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    specs = [("Silhouette", "Silhouette (cao hơn tốt hơn)", "{:.2f}"),
             ("DaviesBouldin", "Davies–Bouldin (thấp hơn tốt hơn)", "{:.2f}"),
             ("CalinskiHarabasz", "Calinski–Harabasz (cao hơn tốt hơn)", None)]
    y = np.arange(len(data))[::-1]
    for ax, (col, title, fmt) in zip(axes, specs):
        ax.barh(y, data[col], color=colors)
        for yi, v in zip(y, data[col]):
            label = fmt_int(v) if fmt is None else fmt.format(v).replace(".", ",")
            ax.text(v, yi, "  " + label, va="center", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, data[col].max() * 1.3)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(4))
        if fmt is None:
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(thousands))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{m} ({c} nhóm)" for m, c in zip(data["Model"], data["Components"])])
    fig.suptitle("So sánh các mô hình trên cùng dữ liệu (K-Means k=4 là mô hình chính)",
                 fontsize=15, fontweight="bold", y=1.03)
    save(fig, out, "06_model_comparison")


# ---------------------------------------------------------------- Phân khúc

def standardized(segments: pd.DataFrame) -> np.ndarray:
    # Giống pipeline: StandardScaler(log1p(RFM)).
    return StandardScaler().fit_transform(np.log1p(segments[RFM]))


def fig_snake(segments: pd.DataFrame, out: Path) -> None:
    x = pd.DataFrame(standardized(segments), columns=RFM)
    x["Segment"] = segments["Segment"].values
    means = x.groupby("Segment")[RFM].mean().loc[SEGMENT_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, row in means.iterrows():
        ax.plot(RFM, row.values, marker="o", lw=2.5, ms=8, color=SEGMENT_COLORS[seg], label=seg)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("Giá trị chuẩn hóa trung bình\n(0 = trung bình toàn bộ khách)")
    ax.set_xticks(range(3), ["Recency\n(thấp = mới mua gần đây)", "Frequency\n(cao = mua nhiều lần)",
                             "Monetary\n(cao = chi nhiều)"])
    ax.set_title("Chân dung 4 nhóm khách hàng theo RFM (snake plot)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, out, "07_segment_snake_plot")


def fig_pca(segments: pd.DataFrame, out: Path) -> None:
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(standardized(segments))
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for seg in SEGMENT_ORDER[::-1]:
        mask = (segments["Segment"] == seg).values
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.55,
                   color=SEGMENT_COLORS[seg], label=f"{seg} ({mask.sum():,})".replace(",", "."))
        cx, cy = coords[mask].mean(axis=0)
        ax.text(cx, cy, seg, fontsize=10, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SEGMENT_COLORS[seg], alpha=0.9))
    ax.set(xlabel=f"PC1 ({ev[0]:.0%} phương sai)", ylabel=f"PC2 ({ev[1]:.0%} phương sai)",
           title="Các nhóm K-Means trên mặt phẳng PCA")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], markerscale=2.5, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, out, "08_segment_pca")


def fig_boxplots(segments: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, RFM):
        data = [segments.loc[segments["Segment"] == s, col] for s in SEGMENT_ORDER]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                        medianprops=dict(color="black", lw=1.5))
        for patch, seg in zip(bp["boxes"], SEGMENT_ORDER):
            patch.set_facecolor(SEGMENT_COLORS[seg])
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, 5), [s.replace(" tiềm", "\ntiềm") for s in SEGMENT_ORDER], fontsize=10)
        ax.set_title(FEATURE_NAMES[col], fontsize=12)
        if col != "Recency":
            ax.set_yscale("log")
            if col == "Frequency":
                ax.set_yticks([1, 2, 3, 5, 10, 20, 30])
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(thousands))
            ax.yaxis.set_minor_formatter(mtick.NullFormatter())
        ax.grid(axis="x", visible=False)
    fig.suptitle("Phân bố R, F, M theo nhóm (ẩn ngoại lệ; F và M dùng thang log)",
                 fontsize=15, fontweight="bold", y=1.02)
    save(fig, out, "09_segment_boxplots")


def fig_profile_table(profiles: pd.DataFrame, out: Path) -> None:
    cols = ["Số khách", "% khách", "% doanh thu", "Recency (ngày)", "Số hóa đơn", "Chi tiêu (£)"]
    cells = [[fmt_int(r.CustomerCount), f"{r.CustomerShare:.0%}", f"{r.MonetaryShare:.0%}",
              fmt_int(r.RecencyMedian), fmt_int(r.FrequencyMedian), fmt_int(r.MonetaryMedian)]
             for r in profiles.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=profiles["Segment"], colLabels=cols,
                     cellLoc="center", rowLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(ACCENT)
        elif c == -1:
            seg = profiles["Segment"].iloc[r - 1]
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(SEGMENT_COLORS[seg])
    fig.suptitle("Tóm tắt 4 nhóm khách hàng (Recency, số hóa đơn, chi tiêu: giá trị trung vị)",
                 fontsize=14, fontweight="bold", y=0.98)
    save(fig, out, "10_segment_profile_table")


# ---------------------------------------------------------------- Slide

def fig_headline(profiles: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = [("Số khách hàng", "CustomerShare"), ("Doanh thu", "MonetaryShare")]
    for i, (label, col) in enumerate(bars):
        left = 0.0
        for r in profiles.itertuples():
            share = getattr(r, col)
            ax.barh(i, share, left=left, color=SEGMENT_COLORS[r.Segment], edgecolor="white", height=0.6)
            if share >= 0.05:
                ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center",
                        color="white", fontsize=15, fontweight="bold")
            left += share
    ax.set_yticks([0, 1], [b[0] for b in bars], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    vip = profiles.set_index("Segment").loc["VIP"]
    sleep = profiles.set_index("Segment").loc["Ngủ đông"]
    fig.suptitle(f"{vip.CustomerShare:.0%} khách VIP mang về {vip.MonetaryShare:.0%} doanh thu",
                 fontsize=24, fontweight="bold", color=SEGMENT_COLORS["VIP"], x=0.125, ha="left", y=1.04)
    ax.set_title(f"Trong khi nhóm Ngủ đông chiếm {sleep.CustomerShare:.0%} số khách nhưng chỉ "
                 f"{sleep.MonetaryShare:.0%} doanh thu", fontsize=13, fontweight="normal",
                 color=MUTED, loc="left", pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s]) for s in SEGMENT_ORDER]
    ax.legend(handles, SEGMENT_ORDER, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=12)
    save(fig, out, "11_slide_headline_value_share")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="Thư mục gốc repo")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Mặc định: <root>/outputs/figures_report")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output_dir or root / "outputs/figures_report").resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    segments, profiles, selection, comparison, audit = load_inputs(root)
    print("[1/3] EDA", flush=True)
    tx = load_transactions(root)
    fig_cleaning_funnel(audit, out)
    fig_monthly_revenue(tx, out)
    fig_countries(tx, out)
    fig_rfm_distributions(segments, out)
    print("[2/3] Model", flush=True)
    fig_kmeans_selection(selection, out)
    fig_model_comparison(comparison, out)
    print("[3/3] Segments + slide", flush=True)
    fig_snake(segments, out)
    fig_pca(segments, out)
    fig_boxplots(segments, out)
    fig_profile_table(profiles, out)
    fig_headline(profiles, out)
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
