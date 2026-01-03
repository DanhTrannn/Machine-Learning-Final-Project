import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.engine import CustomerEngine

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="Hệ thống Phân cụm Khách hàng", layout="wide")

# 2. LOAD MÔ HÌNH
@st.cache_resource
def load_all():
    # Đảm bảo đường dẫn đúng tới các file .pkl của bạn
    engine = joblib.load('models/preprocessor.pkl')
    model = joblib.load('models/best_model.pkl')
    return engine, model

pre, cluster_mod = load_all()

# 3. GIAO DIỆN SIDEBAR (Dự đoán)
with st.sidebar:
    st.header("🔮 Dự đoán khách mới")
    with st.form("input_form"):
        r = st.number_input("Recency (Ngày mua gần nhất)", 0)
        f = st.number_input("Frequency (Số lần mua)", 1)
        m = st.number_input("Monetary (Tổng chi tiêu)", 0.0)
        submit = st.form_submit_button("Dự đoán phân khúc")

# 4. GIAO DIỆN CHÍNH (Báo cáo & Trực quan hóa)
st.title("🛡️ Hệ thống Phân cụm Khách hàng Đa tầng")
st.markdown("---")

# Chia màn hình chính thành các Tabs để trình bày logic khoa học
tab1, tab2, tab3 = st.tabs(["📑 Quy trình đề xuất", "📊 Phân tích tối ưu (K)", "👥 Đặc trưng phân khúc"])

with tab1:
    st.subheader("Quy trình Phân cụm Đa tầng tự động")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**Bước 1: PCA**\n\nKhử nhiễu và nén dữ liệu RFM. Giữ lại >90% phương sai để giải quyết đa cộng tuyến.")
    with col_b:
        st.info("**Bước 2: Đa thuật toán**\n\nSo sánh K-Means, DBSCAN và Hierarchical để tìm cấu trúc cụm phù hợp nhất.")
    with col_c:
        st.info("**Bước 3: Tối ưu**\n\nDùng Silhouette Score để chọn mô hình có độ tách biệt cụm cao nhất.")

with tab2:
    st.subheader("Kết quả thực nghiệm tìm K tối ưu")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**1. Phương pháp Khuỷu tay (Elbow Method)**")
        # Giả lập biểu đồ Elbow (Bạn có thể thay bằng dữ liệu thật từ quá trình train)
        ks = np.arange(1, 11)
        wcss = [100, 45, 25, 15, 12, 10, 8, 7, 6, 5] # Giả lập dữ liệu WCSS
        fig1, ax1 = plt.subplots()
        ax1.plot(ks, wcss, marker='o', linestyle='--', color='red')
        ax1.axvline(x=4, color='blue', linestyle='--') # Giả sử K=4 là tối ưu
        ax1.set_xlabel("Số lượng cụm K")
        ax1.set_ylabel("WCSS (Tổng bình phương khoảng cách)")
        st.pyplot(fig1)
        st.caption("Điểm khuỷu tay xuất hiện tại K=4, cho thấy sự bão hòa của dữ liệu.")

    with c2:
        st.write("**2. Chỉ số Silhouette (Silhouette Score)**")
        # Giả lập biểu đồ Silhouette
        models = ['K-Means', 'Agglomerative', 'DBSCAN']
        scores = [0.65, 0.61, 0.38]
        fig2, ax2 = plt.subplots()
        sns.barplot(x=models, y=scores, palette='viridis', ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Silhouette Score")
        st.pyplot(fig2)
        st.caption("K-Means đạt điểm cao nhất (0.65), cho thấy các cụm tách biệt rõ ràng nhất.")

with tab3:
    st.subheader("Giải mã đặc trưng các nhóm khách hàng")
    # Bảng mô tả ý nghĩa các cụm (Dựa trên phân tích ở Chương 4 của bạn)
    data = {
        "Nhóm": ["Cụm 0", "Cụm 1", "Cụm 2", "Cụm 3"],
        "Đặc điểm": ["VIP", "Tiềm năng", "Nguy cơ rời bỏ", "Vãng lai"],
        "Hành vi": ["Mua thường xuyên, chi đậm", "Mới mua, đang phát triển", "Đã từng mua nhiều nhưng nghỉ lâu", "Ít mua, giá trị thấp"],
        "Đề xuất": ["Tri ân, đặc quyền", "Gửi khuyến mãi lần 2", "Email Marketing kéo lại", "Theo dõi thêm"]
    }
    st.table(pd.DataFrame(data))

# 5. XỬ LÝ DỰ ĐOÁN (In kết quả đè lên phần dưới cùng hoặc giữa)
if submit:
    input_df = pd.DataFrame([[r, f, m]], columns=['Recency', 'Frequency', 'Monetary'])
    
    # Thực hiện quy trình Pipeline
    X_scaled = pre.transform_data(input_df, is_train=False)
    X_pca = pre.pca.transform(X_scaled)
    res = cluster_mod.predict(X_pca)[0]
    
    st.markdown("---")
    st.header(f"🎯 Kết quả dự đoán: **Cụm {res}**")
    
    # Hiển thị tư vấn dựa trên cụm
    if res == 0:
        st.success("💎 **Đây là Khách hàng VIP:** Hãy áp dụng chế độ chăm sóc khách hàng ưu tiên.")
    elif res == 1:
        st.info("📈 **Đây là Khách hàng Tiềm năng:** Hãy khuyến khích họ mua thêm để trở thành VIP.")
    elif res == 2:
        st.warning("⚠️ **Khách hàng Nguy cơ rời bỏ:** Cần có chương trình giảm giá để lôi kéo họ quay lại.")
    else:
        st.error("💤 **Khách hàng Vãng lai:** Nhóm này ít tương tác, không nên tập trung quá nhiều chi phí.")