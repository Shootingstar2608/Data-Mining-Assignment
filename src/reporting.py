"""Generate the Vietnamese report directly from saved results."""

import json
import os
from pathlib import Path
import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.4f}"
        return str(v)
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def write_report(output: Path, audit: dict, path: Path) -> None:
    tables = output / "tables"
    comparison = pd.read_csv(tables / "model_comparison.csv")
    profiles = pd.read_csv(tables / "segment_profiles.csv")
    stability = pd.read_csv(tables / "seed_stability.csv").groupby("Model").ARI.agg(["min", "median"]).reset_index()
    summary = json.loads((output / "run_summary.json").read_text())
    gmm_sensitivity = pd.read_csv(output / "gmm_bic/tables/regularization_sensitivity.csv")
    gmm_stability = json.loads((output / "gmm_bic/robustness_summary.json").read_text())
    text = f"""# Phân khúc khách hàng Online Retail

Báo cáo được tạo tự động bằng `python src/pipeline.py`. Nguồn số liệu hiện hành
là `outputs/tables/` và `outputs/run_summary.json`; các bản PR lưu trong `results/`
và notebook snapshot chỉ dùng tham khảo lịch sử.

## 1. Bài toán và phạm vi

Phân nhóm khách hàng theo hành vi mua trong kỳ để đề xuất chăm sóc và marketing.
Đây là học không giám sát: không có nhãn khách hàng chuẩn để tính accuracy.
Kết quả mô tả hành vi lịch sử, chưa chứng minh hiệu quả của chiến dịch.

Dữ liệu gốc: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail),
tệp `Online Retail.xlsx` đi kèm repo. Kỳ quan sát thực tế: **{audit['first_transaction']}**
đến **{audit['last_transaction']}**. Đơn vị tiền của dữ liệu là bảng Anh (GBP).

## 2. Làm sạch và định nghĩa đặc trưng

{markdown_table(pd.DataFrame(list(audit['counts'].items()), columns=['Chỉ tiêu', 'Số lượng']))}

Loại dòng thiếu CustomerID, hóa đơn hủy khỏi bảng mua, số lượng/giá không dương,
dòng trùng và mã ngoài quy tắc sản phẩm `^\\d{{5}}`. Quy tắc mã hàng là giả định
làm sạch của project, cần rà soát lại khi áp dụng cho nguồn dữ liệu khác.
Giao dịch hoàn/hủy lấy từ dữ liệu gốc: mã hóa đơn C, số lượng âm, giá dương,
cùng quy tắc sản phẩm và loại trùng.

- **Recency:** số ngày từ lần mua gần nhất đến thời điểm mua hợp lệ cuối + 1 ngày.
- **Frequency:** số hóa đơn mua hợp lệ khác nhau; chưa giảm khi hoàn một phần đơn.
- **Monetary:** tổng giá trị mua hợp lệ trừ giá trị hoàn/hủy ghi nhận trong kỳ,
  làm tròn hai chữ số; giữ khách có giá trị cuối dương.
- Tenure, AvgOrderValue, UniqueProducts, TotalItems, ReturnRate và Country dùng
  mô tả nhóm, không đưa vào ma trận phân cụm chính.

Monetary là **chi tiêu ròng ghi nhận trong kỳ**, không phải lợi nhuận. Không có
đối soát từng dòng hoàn với hóa đơn mua; hoàn của đơn trước kỳ vẫn có thể bị trừ.
Recency/Frequency mô tả hoạt động mua, không khẳng định khách đã giữ lại sản phẩm.

## 3. Khám phá và tiền xử lý

RFM có phân phối lệch; dùng `log1p` rồi `StandardScaler` để cân bằng thang đo.
Cả mô hình và metrics chính dùng **cùng {summary['customer_count']:,} khách, cùng thứ tự ID,
cùng ba đặc trưng và không chặn ngoại lệ**. Tất cả dữ liệu được dùng để mô tả
phân khúc hiện tại; đây không phải đánh giá khả năng dự báo tương lai.

![Phân phối](../outputs/figures/rfm_distributions.png)

RFM scoring là baseline quy tắc. Điểm theo ngưỡng phân vị thực nghiệm, các giá trị
bằng nhau nhận cùng điểm; vì thế số khách mỗi mức điểm không nhất thiết bằng nhau.

## 4. Mô hình và cách chọn

K-Means khảo sát k=2–10, xét inertia, silhouette, Davies–Bouldin, độ ổn định
và khả năng diễn giải. k=4 là cấu hình kinh doanh được chọn để có bốn nhóm hành động;
không khẳng định k=4 tối ưu tuyệt đối trên mọi chỉ số.

GMM k=4 dùng full covariance, reg_covar=0.1, 30 lần khởi tạo. Ward dùng k=4.
GMM BIC khảo sát 2–10 thành phần và bốn dạng covariance; chỉ nhận ứng viên hội tụ,
có nhóm nhỏ nhất ít nhất max(20, 1% số khách). BIC dùng chọn mô hình mật độ GMM,
không so trực tiếp với inertia K-Means. Ma trận, cấu hình và mô hình đều được lưu.

![Chọn k](../outputs/figures/kmeans_selection.png)

## 5. Kết quả so sánh

Silhouette, Davies–Bouldin và Calinski–Harabasz dưới đây tính trên **toàn bộ**
ma trận chung. Silhouette càng cao, DB càng thấp, CH càng cao càng thuận lợi
theo tiêu chí tương ứng; các chỉ số này không đo thành công kinh doanh.

{markdown_table(comparison)}

![So sánh](../outputs/figures/model_comparison.png)

Độ ổn định qua năm seed (0–4), tính ARI cho mọi cặp, không phụ thuộc hoán vị nhãn:

{markdown_table(stability)}

Độ nhạy GMM BIC theo regularization trên cùng đầu vào không chặn:

{markdown_table(gmm_sensitivity)}

Với cấu hình BIC chọn và 10 lần khởi tạo, năm seed cho ARI cặp thấp nhất
**{gmm_stability['minimum_pairwise_ari']:.4f}**, trung vị
**{gmm_stability['median_pairwise_ari']:.4f}**. Đây là kiểm tra khởi tạo riêng;
mô hình GMM BIC cuối dùng 30 lần khởi tạo. Số nhóm BIC chọn nhạy với regularization,
vì vậy không tự động dùng số nhóm này làm phân khúc kinh doanh cuối.

K-Means k=4 được giữ làm mô hình chính nhờ sự cân bằng giữa chất lượng phân cụm,
kích thước nhóm, độ ổn định và khả năng diễn giải. GMM k=4 là đối chiếu trực tiếp;
GMM BIC là phân tích khám phá ở số nhóm khác. RFM scoring có số nhóm khác nên
metrics chỉ cung cấp ngữ cảnh, không xem là cuộc so sánh cùng độ chi tiết.

## 6. Chân dung và hành động

{markdown_table(profiles[['Segment', 'CustomerCount', 'RecencyMedian', 'FrequencyMedian', 'MonetaryMedian', 'TenureMedian', 'CustomerShare', 'MonetaryShare']])}

![PCA](../outputs/figures/kmeans_pca.png)

![Tỷ trọng](../outputs/figures/customer_value_shares.png)

Tên VIP, Ngủ đông, Khách mới và Trung thành tiềm năng là nhãn mô tả tương đối
trong tập dữ liệu. "Khách mới" là nhóm có tenure ngắn hơn nhóm trung gian còn lại,
không đảm bảo từng khách đều mới mua lần đầu. Các tên trong CSV viết không dấu.

{markdown_table(pd.read_csv(tables / 'segment_strategies.csv'))}

Các hành động là giả thuyết. Cần thử nghiệm có nhóm đối chứng, kiểm soát chi phí,
biên lợi nhuận và đo KPI trước khi triển khai diện rộng. GMM_Confidence là xác suất
hậu nghiệm theo GMM, không phải độ chắc chắn của nhãn K-Means hoặc nhãn marketing.

## 7. Giới hạn và bước tiếp theo

- Dữ liệu lịch sử của một nhà bán lẻ; không suy rộng trực tiếp cho doanh nghiệp khác.
- Nhiều khách Frequency=1; cấu trúc rời rạc và ngoại lệ ảnh hưởng GMM.
- Tên nhóm và k=4 có yếu tố phán đoán; ranh giới nhóm không phải sự thật cố định.
- Kiểm tra seed đánh giá khởi tạo, chưa thay thế bootstrap và kiểm tra theo thời gian.
- Để triển khai: đánh giá trên kỳ sau, theo dõi drift và thay đổi nhóm; RFM cho khách
  mới phải dùng cùng cửa sổ thời gian và định nghĩa nghiệp vụ trước khi suy luận.

`outputs/tables/preprocessing_sensitivity.csv` kiểm tra riêng tác động chặn phân vị 99%
đối với K-Means k=4. Các kết quả GMM chặn 99% từ PR cũ được giữ riêng trong `results/`;
không trộn các metrics khác ma trận vào bảng so sánh chính.

{markdown_table(pd.read_csv(tables / 'preprocessing_sensitivity.csv'))}

## 8. Tái lập và xác minh

Chạy `python src/pipeline.py`, sau đó `python src/verify_project.py`.
Fingerprint CSV chuẩn hóa LF/CRLF, vẫn phát hiện thay đổi nội dung.
`outputs/run_summary.json` ghi phiên bản thư viện; `requirements-lock.txt` khóa
môi trường đã kiểm tra. Model lưu bằng joblib chỉ được nạp từ nguồn tin cậy.

Nguồn dữ liệu: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning
Repository. [DOI](https://doi.org/10.24432/C5BW33),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Các bảng đặc trưng và
kết quả của project là bản biến đổi từ dữ liệu gốc.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    figures_relative = Path(os.path.relpath(output / "figures", path.parent)).as_posix()
    text = text.replace("../outputs/figures", figures_relative)
    path.write_text(text, encoding="utf-8")
