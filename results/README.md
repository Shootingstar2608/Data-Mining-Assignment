# Kết quả lịch sử từ hai PR

**Kết quả hiện hành ở `outputs/`; đọc `docs/REPORT.md`.** Thư mục này bảo tồn
các kết quả trước tích hợp để truy vết, không dùng làm báo cáo mới nhất.

| Thư mục | Nguồn và ý nghĩa |
| --- | --- |
| `pr1_snapshot/` | Outputs nguyên bản PR #1: notebook dùng GMM n_init=5, baseline RFM chia tie theo thứ tự dòng |
| `gmm_regularized/` | PR #2: RFM gross 4.338 khách, chặn 99% |
| `gmm_group_features/` | PR #2: features 4.320 khách, chặn 99% |
| `gmm_shared_matrix/` | PR #2: features 4.320 khách, không chặn |
| `public_model_comparison/` | PR #2: đối chiếu K-Means/GMM trên ma trận chung |

Các model lịch sử được tạo bằng scikit-learn 1.7.2; môi trường tích hợp hiện
được khóa riêng. `verify_gmm.py` hỗ trợ fingerprint CRLF của bản cũ.
