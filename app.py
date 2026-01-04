import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- CẤU HÌNH TRANG (Chuẩn doanh nghiệp) ---
st.set_page_config(
    page_title="Customer Segmentation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH (Giao diện phẳng, hiện đại, không icon) ---
st.markdown("""
<style>
    /* Font toàn hệ thống */
    .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Header chính */
    .main-header {
        font-size: 32px;
        font-weight: 700;
        color: #0D47A1; /* Deep Blue */
        margin-bottom: 10px;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 10px;
    }
    
    /* Sub-header */
    .sub-header {
        font-size: 20px;
        font-weight: 600;
        color: #424242;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Metric Card Style (Tạo khối thẻ bài) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Tùy chỉnh Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        border: none;
        background-color: transparent;
        color: #757575;
    }
    .stTabs [aria-selected="true"] {
        color: #0D47A1;
        border-bottom: 3px solid #0D47A1;
    }
    
    /* Alert/Info Box */
    .info-box {
        padding: 15px;
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        border-radius: 4px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM LOAD DATA ---
@st.cache_data
def load_data(path: str):
    if not os.path.exists(path):
        return None, None
    try:
        artifact = joblib.load(path)
        data = pd.DataFrame(artifact["data"])
        # Ép kiểu string cho ID để tránh lỗi hiển thị số
        if 'CustomerID' in data.columns:
            data['CustomerID'] = data['CustomerID'].astype(str)
        metadata = artifact.get("metadata", {})
        return data, metadata
    except Exception:
        return None, None

# --- HÀM EXPERIMENT K-MEANS ---
@st.cache_data
def run_kmeans_experiment(df_values, k_range):
    distortions = []
    silhouettes = []
    
    # Sampling nếu dữ liệu lớn để tối ưu hiệu năng
    if len(df_values) > 2000:
        indices = np.random.choice(len(df_values), 2000, replace=False)
        X_sample = df_values[indices]
    else:
        X_sample = df_values

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_sample)
        distortions.append(model.inertia_)
        if k > 1:
            score = silhouette_score(X_sample, model.labels_)
            silhouettes.append(score)
        else:
            silhouettes.append(0)
    return distortions, silhouettes

# --- HÀM BUSINESS LOGIC (Không dùng icon) ---
def get_marketing_action(r, f, m, r_mean, f_mean, m_mean):
    score = 0
    if r < r_mean: score += 1
    if f > f_mean: score += 1
    if m > m_mean: score += 1
    
    # Trả về Tuple: (Loại khách, Màu sắc đại diện, Hành động)
    if score == 3:
        return "VIP Customer", "green", "Prioritize premium support & Exclusive offers."
    elif score == 0:
        return "Hibernating", "red", "Re-activation campaigns with high discounts."
    elif m > m_mean:
        return "Big Spender", "blue", "Cross-sell high-value items."
    elif f > f_mean:
        return "Loyal Customer", "orange", "Loyalty programs & Membership tiers."
    else:
        return "Standard Potential", "grey", "Monitor behavior & Standard newsletters."

# --- MAIN APP ---
def main():
    st.markdown('<div class="main-header">Ensemble Customer Segmentation</div>', unsafe_allow_html=True)

    # 1. Load Data
    pkl_path = "artifacts/result_dashboard.pkl"
    if not os.path.exists(pkl_path):
        st.error("Data file not found. Please run the notebook to generate artifacts.")
        st.stop()

    df, metadata = load_data(pkl_path)
    
    cluster_col = metadata.get("cluster_col", "Cluster_Random_Ensemble")
    # Fallback tìm cột cluster
    if cluster_col not in df.columns:
        cols = [c for c in df.columns if 'cluster' in c.lower()]
        cluster_col = cols[0] if cols else None

    if not cluster_col:
        st.error("Cluster column not found in dataset.")
        st.stop()

    # --- SIDEBAR (Clean UI) ---
    with st.sidebar:
        st.markdown("### Filter Settings")
        
        all_clusters = sorted(df[cluster_col].unique())
        selected_clusters = st.multiselect(
            "Select Clusters", 
            all_clusters, 
            default=all_clusters
        )
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.text(f"Algorithm: {metadata.get('model_name', 'Hybrid Ensemble')}")
        st.text(f"Total Records: {len(df):,}")
        st.markdown("---")
        
        if st.button("Reset Cache"):
            st.cache_data.clear()

    if not selected_clusters:
        st.warning("Please select at least one cluster to view data.")
        st.stop()

    # Filter Data
    df_filtered = df[df[cluster_col].isin(selected_clusters)]
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    valid_cols = [c for c in rfm_cols if c in df.columns]

    # --- TABS LAYOUT ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard Overview", 
        "3D Analysis", 
        "Business Recommendations", 
        "Customer Search",
        "Model Diagnostics"
    ])

    # === TAB 1: OVERVIEW ===
    with tab1:
        st.markdown('<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)
        
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Customers", f"{len(df_filtered):,}")
        m2.metric("Avg. Revenue", f"${df_filtered['Monetary'].mean():,.2f}")
        m3.metric("Avg. Frequency", f"{df_filtered['Frequency'].mean():.2f}")
        m4.metric("Avg. Recency", f"{df_filtered['Recency'].mean():.1f} days")
        
        st.markdown("---")
        
        # Charts Area
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Cluster Distribution**")
            fig_pie = px.pie(
                df_filtered, 
                names=cluster_col, 
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.markdown("**Cluster Characteristics (Radar Chart)**")
            if len(valid_cols) == 3:
                # Normalize data for Radar
                df_norm = df[df[cluster_col].isin(selected_clusters)].copy()
                for c in valid_cols:
                    df_norm[c] = (df_norm[c] - df[c].min()) / (df[c].max() - df[c].min())
                
                radar_data = df_norm.groupby(cluster_col)[valid_cols].mean().reset_index()
                
                fig_radar = go.Figure()
                for i, row in radar_data.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row[valid_cols].values,
                        theta=valid_cols,
                        fill='toself',
                        name=f'Cluster {row[cluster_col]}'
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=350, margin=dict(t=20, b=20))
                st.plotly_chart(fig_radar, use_container_width=True)

    # === TAB 2: 3D ANALYSIS ===
    with tab2:
        st.markdown('<div class="sub-header">Multidimensional Visualization</div>', unsafe_allow_html=True)
        if len(valid_cols) == 3:
            fig_3d = px.scatter_3d(
                df_filtered, 
                x='Recency', y='Frequency', z='Monetary',
                color=cluster_col,
                opacity=0.6,
                height=600,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # === TAB 3: RECOMMENDATIONS ===
    with tab3:
        st.markdown('<div class="sub-header">Strategic Actions</div>', unsafe_allow_html=True)
        
        global_mean = df[valid_cols].mean()
        
        # Grid layout for recommendations
        cols = st.columns(2)
        
        for idx, c_id in enumerate(selected_clusters):
            with cols[idx % 2]:
                with st.container():
                    # Tính toán logic
                    c_data = df[df[cluster_col] == c_id]
                    avg = c_data[valid_cols].mean()
                    lbl, color, advice = get_marketing_action(
                        avg['Recency'], avg['Frequency'], avg['Monetary'],
                        global_mean['Recency'], global_mean['Frequency'], global_mean['Monetary']
                    )
                    
                    # Custom HTML Card
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:15px; border-radius:5px; margin-bottom:20px; border-left: 5px solid {color}">
                        <h4 style="margin:0; color: #333;">Cluster {c_id}: {lbl}</h4>
                        <p style="margin-top:10px; font-size:14px; color:#555;"><i>Strategy: {advice}</i></p>
                        <hr style="border-top: 1px dashed #eee;">
                        <small>Avg M: ${avg['Monetary']:,.0f} | Avg F: {avg['Frequency']:.1f} | Avg R: {avg['Recency']:.1f}</small>
                    </div>
                    """, unsafe_allow_html=True)

    # === TAB 4: SEARCH ===
    with tab4:
        st.markdown('<div class="sub-header">Customer Lookup</div>', unsafe_allow_html=True)
        
        search_col, res_col = st.columns([1, 3])
        with search_col:
            cid = st.text_input("Enter Customer ID", placeholder="Type ID here...")
        
        with res_col:
            if cid and 'CustomerID' in df.columns:
                res = df[df['CustomerID'] == cid]
                if not res.empty:
                    st.success(f"Customer Found")
                    st.dataframe(res, use_container_width=True)
                else:
                    st.warning("No customer found with this ID.")

    # === TAB 5: DIAGNOSTICS ===
    with tab5:
        st.markdown('<div class="sub-header">Model Training & Validation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Elbow Method Simulation**")
            st.write("Run real-time K-Means on sample data to verify cluster count.")
            k_max = st.slider("Max K", 3, 10, 6)
            if st.button("Run Simulation"):
                with st.spinner("Processing..."):
                     # Log transform logic for stability
                    X_lab = np.log1p(df[valid_cols].values)
                    dist, sil = run_kmeans_experiment(X_lab, range(1, k_max + 1))
                    
                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(go.Scatter(x=list(range(1, k_max+1)), y=dist, name='Inertia'))
                    fig_elbow.update_layout(title="Inertia vs K", height=300, margin=dict(l=20, r=20, t=30, b=20))
                    
                    # Store in session state to persist
                    st.session_state['fig_elbow'] = fig_elbow
        
        with col2:
            if 'fig_elbow' in st.session_state:
                st.plotly_chart(st.session_state['fig_elbow'], use_container_width=True)
            else:
                st.info("Click 'Run Simulation' to see the chart.")

        st.markdown("---")
        st.markdown("**PCA Visualization (2D Projection)**")
        if st.checkbox("Show PCA"):
             # Simple PCA
            X_pca = np.log1p(df[valid_cols].values)
            pca = PCA(n_components=2)
            comps = pca.fit_transform(X_pca)
            pca_df = pd.DataFrame(comps, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = df[cluster_col].values
            
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', opacity=0.5, 
                                color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig_pca, use_container_width=True)

if __name__ == "__main__":
    main()