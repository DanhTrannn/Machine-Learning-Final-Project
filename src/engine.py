import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

class CustomerEngine:
    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.90)
        
    def clean_and_rfm(self, df):
        # 1. Ép kiểu
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['CustomerID'] = df['CustomerID'].astype(str)
        
        # 2. Xử lý Missing (Quy tắc 5%)
        missing_pct = df.isnull().sum() / len(df)
        for col in df.columns:
            if missing_pct[col] > 0:
                if missing_pct[col] < 0.05:
                    df = df.dropna(subset=[col])
                else:
                    fill_val = df[col].mode()[0] if df[col].dtype == 'object' else df[col].median()
                    df[col] = df[col].fillna(fill_val)
        
        # 3. Tính RFM
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        df['TotalSum'] = df['Quantity'] * df['UnitPrice']
        snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)
        
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot - x.max()).days,
            'InvoiceNo': 'count',
            'TotalSum': 'sum'
        }).rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalSum':'Monetary'})
        
        # 4. Outlier (IQR)
        for col in rfm.columns:
            Q1, Q3 = rfm[col].quantile(0.25), rfm[col].quantile(0.75)
            IQR = Q3 - Q1
            rfm = rfm[(rfm[col] >= Q1 - 1.5*IQR) & (rfm[col] <= Q3 + 1.5*IQR)]
        return rfm

    def transform_data(self, rfm_df, is_train=True):
        if is_train:
            X_trans = self.pt.fit_transform(rfm_df)
            X_scaled = self.scaler.fit_transform(X_trans)
        else:
            X_trans = self.pt.transform(rfm_df)
            X_scaled = self.scaler.transform(X_trans)
        return X_scaled

    def build_ensemble(self, X_pca):
        # Tạo nhãn từ 2 model khác nhau để làm Ensemble (Consensus)
        km_labels = KMeans(n_clusters=4, random_state=42).fit_predict(X_pca)
        ac_labels = AgglomerativeClustering(n_clusters=4).fit_predict(X_pca)
        
        # Kết hợp nhãn làm feature mới
        ensemble_X = np.column_stack((km_labels, ac_labels))
        final_labels = AgglomerativeClustering(n_clusters=4).fit_predict(ensemble_X)
        return final_labels