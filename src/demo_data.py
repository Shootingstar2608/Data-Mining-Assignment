"""Read-only data preparation and scenario inference for the Streamlit demo."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from reproducibility import matches_input

RFM = ["Recency", "Frequency", "Monetary"]
SEGMENTS = {
    "VIP": "VIP",
    "Trung thanh tiem nang": "Trung thành tiềm năng",
    "Khach moi": "Khách mới",
    "Ngu dong": "Ngủ đông",
}
COLORS = {"VIP": "#246BCE", "Trung thành tiềm năng": "#168C7F",
          "Khách mới": "#A16BD8", "Ngủ đông": "#DA8B36"}
ACTIONS = {
    "VIP": ("Giữ chân những khách có đóng góp cao", "Quyền lợi thành viên, chăm sóc phù hợp và chương trình giới thiệu khách.", "Tỷ lệ mua lại · giữ chân · biên lợi nhuận"),
    "Trung thành tiềm năng": ("Khuyến khích quay lại và tăng mức gắn bó", "Nhắc mua lại, gợi ý sản phẩm liên quan và thử ưu đãi theo ngưỡng.", "Tần suất mua · chi tiêu ròng · biên lợi nhuận"),
    "Khách mới": ("Xây dựng thói quen mua lần tiếp theo", "Onboarding, hướng dẫn sản phẩm và thử ưu đãi cho lần mua thứ hai.", "Tỷ lệ mua lần hai trong 60 ngày"),
    "Ngủ đông": ("Tái kích hoạt với chi phí phù hợp", "Thử chiến dịch win-back chi phí thấp; điều chỉnh tần suất liên hệ nếu không phản hồi.", "Tỷ lệ quay lại · chi phí tái kích hoạt"),
}


def load_dashboard(root: Path) -> dict:
    output = root / "outputs"
    required = ["run_summary.json", "data_audit.json", "tables/customer_segments.csv",
                "tables/segment_profiles.csv", "tables/model_comparison.csv",
                "tables/kmeans_selection.csv", "tables/seed_stability.csv",
                "tables/preprocessing_sensitivity.csv", "tables/model_agreement.csv",
                "gmm_bic/run_summary.json", "gmm_bic/tables/regularization_sensitivity.csv"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("Thiếu kết quả: " + ", ".join(missing) + ". Chạy python src/pipeline.py trước.")
    summary = json.loads((output / "run_summary.json").read_text())
    input_path = root / "data/customer_features.csv"
    if not input_path.is_file() or not matches_input(input_path, summary):
        raise ValueError("Dữ liệu đặc trưng không khớp lần chạy đã lưu. Chạy lại python src/pipeline.py.")
    customers = pd.read_csv(output / "tables/customer_segments.csv")
    features = pd.read_csv(input_path).sort_values("CustomerID").reset_index(drop=True)
    if len(customers) != summary["customer_count"] or not customers.CustomerID.is_unique:
        raise ValueError("Bảng khách hàng không khớp số lượng hoặc có CustomerID trùng.")
    pd.testing.assert_frame_equal(features, customers[features.columns], check_dtype=False)
    customers["CustomerID"] = customers.CustomerID.astype(int).astype(str)
    customers["Phân khúc"] = customers.Segment.map(SEGMENTS)
    if customers["Phân khúc"].isna().any():
        raise ValueError("Có tên phân khúc chưa được hỗ trợ trong demo.")
    profiles = pd.read_csv(output / "tables/segment_profiles.csv")
    profiles["Phân khúc"] = profiles.Segment.map(SEGMENTS)
    x = StandardScaler().fit_transform(np.log1p(customers[RFM]))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)
    customers["PC1"], customers["PC2"] = coords[:, 0], coords[:, 1]
    return {
        "customers": customers, "profiles": profiles, "summary": summary,
        "audit": json.loads((output / "data_audit.json").read_text()),
        "comparison": pd.read_csv(output / "tables/model_comparison.csv"),
        "selection": pd.read_csv(output / "tables/kmeans_selection.csv"),
        "stability": pd.read_csv(output / "tables/seed_stability.csv"),
        "sensitivity": pd.read_csv(output / "tables/preprocessing_sensitivity.csv"),
        "agreement": pd.read_csv(output / "tables/model_agreement.csv"),
        "gmm_sensitivity": pd.read_csv(output / "gmm_bic/tables/regularization_sensitivity.csv"),
        "gmm_summary": json.loads((output / "gmm_bic/run_summary.json").read_text()),
        "pca_variance": pca.explained_variance_ratio_.tolist(),
    }


def filter_customers(customers: pd.DataFrame, segments: list, country: str = "Tất cả") -> pd.DataFrame:
    mask = customers["Phân khúc"].isin(segments)
    if country != "Tất cả":
        mask &= customers.Country.eq(country)
    return customers.loc[mask].copy()


def load_models(root: Path) -> tuple[dict, dict]:
    # Only repository-owned bundles are accepted. The demo has no pickle uploader.
    output = root / "outputs/model"
    km = joblib.load(output / "kmeans.joblib")
    if not matches_input(root / "data/customer_features.csv", km):
        raise ValueError("Model K-Means không khớp dữ liệu hiện hành.")
    return km, joblib.load(output / "gmm_k4.joblib")


def predict_scenario(values: list[float], bundles: tuple[dict, dict]) -> dict:
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("R, F và M phải là ba số hữu hạn, lớn hơn 0.")
    km, gm = bundles
    frame = pd.DataFrame([values], columns=RFM)
    with threadpool_limits(limits=1):
        x = km["scaler"].transform(np.log1p(frame))
        raw = int(km["model"].predict(x)[0])
        cluster_id = km["raw_cluster_to_segment_id"][raw]
        probabilities = gm["model"].predict_proba(gm["scaler"].transform(np.log1p(frame)))[0]
    segment = SEGMENTS[km["segment_names"][cluster_id]]
    rows = [{"Thành phần": f"GMM {int(mapped) + 1}", "Xác suất": float(probabilities[raw_id])}
            for raw_id, mapped in gm["raw_cluster_to_segment_id"].items()]
    return {"segment": segment, "probabilities": pd.DataFrame(rows).sort_values("Thành phần"),
            "gmm_confidence": float(probabilities.max())}
