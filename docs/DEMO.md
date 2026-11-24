# Kịch bản demo trước lớp · 5–7 phút

## Chuẩn bị

Từ thư mục project, dùng `.venv` hiện có:

```bash
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -m streamlit run streamlit_app.py
```

Trên Windows dùng `.venv\Scripts\Activate.ps1`. Mở `http://localhost:8501`.
Mở thử một lần trước buổi trình bày. Kết quả và model đã có trong repo;
không cần đọc Excel hoặc chạy lại huấn luyện trong lúc demo. App chạy local
và không cần đăng nhập. Nội dung, dữ liệu và model được xử lý trên máy;
Streamlit có thể cần mạng để tải một số tài nguyên frontend tùy phiên bản.

## Người 1 · Giới thiệu, khoảng 1 phút

- Nêu bài toán: từ RFM nhận diện nhóm khách và đề xuất chăm sóc.
- Chỉ ra 4.320 khách và bốn phân khúc K-Means.
- Đọc biểu đồ tỷ trọng khách và chi tiêu: khoảng 15,9% khách VIP đóng góp 64,1%.
- Mở phần nguồn và định nghĩa R/F/M khi cần.

## Người 2 · EDA, khoảng 1 phút

- Chọn Monetary, tắt rồi bật log1p: giải thích chi tiêu lệch phải và vai trò của log.
- Đọc heatmap Spearman: Frequency và Monetary có tương quan dương mạnh.
- Nêu StandardScaler sau log, cùng ba biến cho các mô hình.
- Phân biệt dòng mua và dòng hoàn/hủy được tổng hợp riêng.

## Người 3 · Modeling, khoảng 1–2 phút

- Đổi Silhouette sang DaviesBouldin: K-Means không thắng mọi chỉ số.
- Mở Chọn số cụm, nêu rõ k=2 có silhouette cao hơn; k=4 được chọn để có mức
  chi tiết phục vụ hành động và vẫn ổn định, dễ giải thích.
- Mở độ ổn định và độ nhạy: K-Means ổn định qua seed; số nhóm GMM BIC nhạy
  với regularization. GMM BIC là phương án khám phá.
- Không gọi silhouette hoặc xác suất GMM là accuracy.

## Người 4 · Visualize, khoảng 2 phút

- Rê chuột trên PCA để xem RFM của một khách. Nêu PCA chỉ là phép chiếu để xem;
  phân cụm dùng đủ ba đặc trưng.
- Chỉ chọn VIP hoặc Ngủ đông; thử lọc United Kingdom; tìm CustomerID hoặc tải CSV.
- Mở chân dung nhóm và đọc một hành động cùng KPI. Phần chân dung dùng toàn tập,
  được ghi nhãn riêng với bộ lọc khám phá.
- Nhấn Xem phân khúc ở tình huống mặc định: R=8, F=10, M=3622.95, gần hồ sơ VIP.
- Đổi thành R=177, F=1, M=290.30 rồi nhấn lại để thấy thay đổi nhóm.
- Giải thích đây là suy luận bằng model đã lưu, chưa phải dự báo mua hàng tương lai.

## Khi được hỏi về giới hạn

- Không có ground truth; đánh giá qua chỉ số nội bộ, độ ổn định và khả năng diễn giải.
- Monetary là chi tiêu ròng ghi nhận trong kỳ, không phải lợi nhuận; hoàn/hủy chưa
  được đối soát từng dòng với đơn mua gốc.
- Tên nhóm có tính tương đối. Khách mới không khẳng định từng người mới mua lần đầu.
- Các hành động marketing là giả thuyết cần thử nghiệm có đối chứng.
- Xác suất GMM thuộc hệ nhãn GMM, không phải độ chắc chắn của nhãn K-Means.
