# Phân khúc khách hàng bằng Data Mining

Project tích hợp hai PR: tạo đặc trưng và K-Means (#1), GMM độc lập (#2).
Mục tiêu là mô tả hành vi khách hàng Online Retail và đề xuất các giả thuyết
chăm sóc/marketing. **K-Means k=4 là mô hình chính**; RFM scoring, GMM và Ward
là các phương án đối chiếu.

## Chạy từ đầu

Môi trường đã kiểm tra: **CPython 3.12**, Linux. Từ thư mục gốc repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python src/pipeline.py
```

Trên Windows, dùng `python -m venv .venv` và `.venv\Scripts\Activate.ps1`.
`requirements-lock.txt` khóa các phiên bản đã chạy; `requirements.txt` mô tả
khoảng phiên bản hỗ trợ nhưng môi trường khác vẫn cần kiểm tra lại.

Pipeline đọc `Online Retail.xlsx` một lần, làm sạch, tạo RFM/đặc trưng, chọn GMM
bằng BIC, kiểm tra độ nhạy regularization và seed, so sánh các mô hình trên cùng
ma trận, xuất biểu đồ và báo cáo, rồi xác minh model và kết quả đã lưu. Nếu một
bước lỗi, lệnh trả mã lỗi; chỉ đọc báo cáo mới sau khi có dòng `DONE`.
Không cần Jupyter để chạy pipeline. Thời gian chạy phụ thuộc máy.

## Kết quả hiện hành

- **[Báo cáo tiếng Việt](docs/REPORT.md)** — số liệu được sinh từ bảng kết quả.
- `outputs/tables/customer_segments.csv` — đặc trưng, điểm baseline và nhãn từng mô hình.
- `outputs/tables/model_comparison.csv` — metrics toàn bộ khách trên cùng ma trận.
- `outputs/tables/segment_profiles.csv`, `segment_strategies.csv` — chân dung, hành động và KPI.
- `outputs/tables/seed_stability.csv` — mọi cặp seed cho K-Means/GMM k=4.
- `outputs/tables/preprocessing_sensitivity.csv` — kiểm tra riêng tác động chặn 99%.
- `outputs/figures/` — phân phối, chọn k, PCA, chân dung và biểu đồ so sánh.
- `outputs/model/` — K-Means, GMM k=4 và scaler để kiểm tra lại phép gán nhóm.
- `outputs/gmm_bic/` — mô hình, probabilities, lựa chọn và độ nhạy GMM BIC.
- `outputs/data_audit.json`, `run_summary.json` — thống kê làm sạch, fingerprint và phiên bản.

Lần chạy đã kiểm tra có 4.320 khách, K-Means 4 nhóm và GMM BIC 8 nhóm. Đây là
kết quả của cấu hình hiện tại, không hardcode số khách trong pipeline.

## Kiểm tra

```bash
python -m unittest discover -s tests -v
python src/verify_project.py
```

CI GitHub chạy kiểm tra hồi quy và toàn bộ pipeline từ Excel, rồi tải kết quả
thành artifact. CI chỉ bắt đầu sau khi nhánh được push lên GitHub.
Fingerprint CSV chuẩn hóa LF/CRLF để dùng nhất quán giữa Windows và Linux.
Model joblib chỉ nạp từ nguồn tin cậy và nên dùng cùng phiên bản thư viện đã lưu.

## Notebook

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/modeling.ipynb
```

Notebook hiện hành đọc kết quả của pipeline, không có bản huấn luyện riêng.
`notebooks/modeling_pr1_snapshot.ipynb` và `results/pr1_snapshot/` giữ nguyên
bản PR #1 trước tích hợp. Các kết quả PR #2 trong `results/gmm_*` và tài liệu
tháng 10 là bản lịch sử; **không trộn với kết quả `outputs/` hiện hành**.

## Quy ước phân tích

Đầu vào chính: `data/customer_features.csv`, chi tiêu ròng ghi nhận trong kỳ,
đã loại trùng và mã ngoài quy tắc sản phẩm. Ba mô hình cùng k=4 dùng chung
`StandardScaler(log1p(RFM))`, không chặn ngoại lệ. GMM BIC chọn số thành phần
riêng; độ nhạy chặn 99% được báo cáo riêng. Xem định nghĩa, giới hạn hoàn/hủy,
phương pháp chọn k và cách đọc xác suất trong [báo cáo](docs/REPORT.md).

Các script ETL, features và GMM từ hai PR vẫn có thể chạy riêng. GMM độc lập
mặc định trên `data/rfm.csv` là cấu hình lịch sử khác với pipeline chính.
Khi thay đầu vào dùng `python src/pipeline.py --help`; cần đủ khách để các
ứng viên GMM đạt điều kiện kích thước nhóm tối thiểu.

## Nguồn dữ liệu

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
[DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33).
[Dữ liệu và mô tả gốc](https://archive.ics.uci.edu/dataset/352/online+retail),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Các bảng trong `data/` và `outputs/` là dữ liệu đã biến đổi từ bộ gốc. Project
này không sử dụng dữ liệu Bibica. Đề xuất kinh doanh cần thử nghiệm thực tế.
