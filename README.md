

---

# 🛒 Customer Segmentation System using Advanced Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![HCMUTE](https://img.shields.io/badge/University-HCMUTE-red.svg)](https://hcmute.edu.vn/)

Dự án này tập trung vào việc phân khúc khách hàng dựa trên hành vi mua sắm từ dữ liệu hóa đơn (Online Retail Dataset). Chúng tôi triển khai từ các phương pháp cơ sở đến mô hình lai đề xuất tự động tối ưu hóa nhằm giúp doanh nghiệp thực hiện chiến dịch Marketing cá nhân hóa.

## 📖 Tổng quan Đồ án
*   **Môn học:** Học Máy (Machine Learning)
## 👥 Thành viên thực hiện (Nhóm 2)
| STT | Họ và Tên | MSSV |
| :--- | :--- | :--- |
| 1 | Phan Trọng Phú | 23133056 |
| 2 | Phan Trọng Quí | 23133061 |
| 3 | Trần Thành Danh | 23133010 |
| 4 | Lê Đăng Khoa | 23133036 |
*   **Giảng viên hướng dẫn:** Ts. Phan Thị Huyền Trang
*   **Năm hoàn thành:** 01/2026

## 🚀 Tính năng chính
- **Tiền xử lý nâng cao:** Xử lý Outliers (IQR), Missing data, Biến đổi phân phối (Box-Cox, Log Transform).
- **RFM Analysis:** Trích xuất đặc trưng hành vi khách hàng: Recency, Frequency, Monetary.
- **Đa thuật toán:** So sánh thực nghiệm giữa K-Means, DBSCAN, Hierarchical Clustering và Bagging Ensemble.
- **Mô hình đề xuất (AutoMultiStageClustering):** Hệ thống tự động giảm chiều dữ liệu (PCA), sàng lọc thuật toán (Lazy Screening) và tinh chỉnh siêu tham số (Auto-tuning).

## 🛠 Quy trình thực hiện (Methodology)

### 1. Feature Engineering (RFM)
Dữ liệu thô được chuyển đổi thành 3 chỉ số hành vi:
- **Recency (R):** Độ tươi mới của lần mua hàng cuối.
- **Frequency (F):** Tần suất giao dịch.
- **Monetary (M):** Tổng giá trị chi tiêu.

### 2. Mô hình Đề xuất: AutoMultiStageClustering
Mô hình lai được thiết kế gồm 3 giai đoạn chính:
1.  **PCA Transformation:** Nén không gian RFM xuống các thành phần chính yếu, khử nhiễu đa cộng tuyến (>90% variance).
2.  **Lazy Screening:** Chạy đua giữa các thuật toán đơn lẻ để chọn ra mô hình tiềm năng nhất dựa trên *Silhouette Score*.
3.  **Hyperparameter Optimization:** Tự động tinh chỉnh tham số chuyên sâu ($K$ tối ưu, $Eps$ tối ưu) bằng các kỹ thuật toán học (Elbow, Silhouette, K-Distance).

## 📊 Kết quả thực nghiệm
Mô hình đề xuất cho kết quả vượt trội so với các phương pháp truyền thống:

| Phương pháp | Silhouette Score | DB Index |
| :--- | :---: | :---: |
| **Proposed Model (Full)** | **0.559** | **0.289** |
| K-Means + RFM | 0.437 | 0.866 |
| DBSCAN (loại nhiễu) | 0.250 | 1.426 |
| Hierarchical | 0.419 | 0.543 |

> **Nhận xét:** Mô hình đề xuất đạt độ tách biệt cụm sắc nét nhất (Silhouette cao nhất) và độ đậm đặc cụm tốt nhất (DB Index thấp nhất).

## 🧪 Nghiên cứu cắt bỏ (Ablation Study)
Chúng tôi thực hiện cắt bỏ từng thành phần của mô hình để kiểm chứng:
- **No-PCA:** Silhouette sụt giảm do nhiễu đa cộng tuyến.
- **No-Tuning:** Silhouette giảm sâu nhất xuống 0.36, khẳng định tầm quan trọng của việc tối ưu siêu tham số.

## 💻 Cài đặt & Sử dụng
1. Clone repository:
   ```bash
   git clone https://github.com/username/customer-segmentation-hcmute.git
   ```
2. Cài đặt thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy lệnh sau để khởi chạy giao diện:
```Bash
streamlit run app.py
```


## ⚖️ License
Dự án được thực hiện cho mục đích học tập tại HCMUTE. Vui lòng trích dẫn nguồn nếu sử dụng lại tài liệu.

---
*Cảm ơn Cô Phan Thị Huyền Trang đã hướng dẫn chúng em hoàn thành đồ án này!*
