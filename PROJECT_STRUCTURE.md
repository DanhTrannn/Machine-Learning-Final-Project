MÔ TẢ CẤU TRÚC DỰ ÁN ML DEPLOY BẰNG STREAMLIT

📂 data/
raw/: dữ liệu gốc ban đầu
processed/: dữ liệu đã làm sạch, xử lý để train model 

📂 notebooks/
EDA, trực quan dữ liệu
Thử nhiều mô hình, không dùng để deploy

📂 src/
Xử lý dữ liệu
Train model
Đánh giá mô hình (CV, metrics)

📂 models/
File .pkl / .joblib
Được load trực tiếp khi chạy app

📂 app/
Nhận input người dùng
Load model
Dự đoán và hiển thị kết quả