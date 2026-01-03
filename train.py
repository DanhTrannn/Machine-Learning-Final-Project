import pandas as pd
import numpy as np
import datetime as dt
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from google.colab import files

# --- 1. ĐỊNH NGHĨA BỘ NÃO XỬ LÝ ---
class CustomerEngine:
    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.90)

    def clean_and_rfm(self, df):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['CustomerID'] = df['CustomerID'].astype(str)
        # Quy tắc xử lý missing 5%
        missing_pct = df.isnull().sum() / len(df)
        for col in df.columns:
            if missing_pct[col] > 0:
                if missing_pct[col] < 0.05: df = df.dropna(subset=[col])
                else: df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())
        # Tính RFM
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        df['TotalSum'] = df['Quantity'] * df['UnitPrice']
        snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot - x.max()).days,
            'InvoiceNo': 'count', 'TotalSum': 'sum'
        }).rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalSum':'Monetary'})
        # Outlier IQR
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

# --- 2. THỰC THI HUẤN LUYỆN ---
# Giả sử bạn đã upload file data.csv lên Colab
df = pd.read_excel('/content/data.xlsx')

engine = CustomerEngine()
rfm = engine.clean_and_rfm(df)
X_scaled = engine.transform_data(rfm)
X_pca = engine.pca.fit_transform(X_scaled)

# Tìm model tốt nhất (Ví dụ K=4)
best_model = KMeans(n_clusters=4, random_state=42).fit(X_pca)

# --- 3. XUẤT FILE ---
joblib.dump(engine, 'preprocessor.pkl')
joblib.dump(best_model, 'best_model.pkl')

files.download('preprocessor.pkl')
files.download('best_model.pkl')