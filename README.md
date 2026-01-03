# 🚀 Machine Learning App with Streamlit

Dự án hướng dẫn quy trình từ huấn luyện mô hình trên Google Colab đến triển khai ứng dụng dự báo giao diện web.

---

## 🛠 1. Cài đặt thư viện

Đầu tiên, hãy tạo file `requirements.txt` với nội dung sau:

```text
streamlit
pandas
scikit-learn
joblib
matplotlib
seaborn
```

Sau đó, mở Terminal và chạy lệnh:
```bash
pip install -r requirements.txt
```

🧪 2. Huấn luyện mô hình (Google Colab)
Thực hiện các bước sau để chuẩn bị mô hình:

Chạy code huấn luyện trong notebook để tạo ra các file định dạng .pkl.

Tải các file .pkl (ví dụ: model.pkl, scaler.pkl) về máy tính cá nhân.

💻 3. Triển khai ứng dụng (VS Code)
📂 Cấu trúc thư mục chuẩn

Hãy đảm bảo các file mô hình được đặt trong thư mục models/:

project-folder/

├── models/

│   └── (dán các file .pkl vào đây)

├── app.py

└── requirements.txt

⚡ Chạy ứng dụng
Mở Terminal tại thư mục gốc của dự án.

Chạy lệnh sau để khởi chạy giao diện:

```Bash
streamlit run app.py
```
Truy cập địa chỉ http://localhost:8501 trên trình duyệt để xem kết quả.
