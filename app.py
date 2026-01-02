import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisa Belanja & Prediksi Harga",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CATATAN PENTING ---
# Kita MENGHAPUS blok st.markdown CSS manual di sini.
# Dengan begitu, Streamlit akan otomatis menangani warna (Light/Dark).

# --- 2. DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('nota.csv')
    except FileNotFoundError:
        # Data dummy untuk preview jika file tidak ada
        data = {
            'Tanggal': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
            'Toko': ['Toko A', 'Toko B', 'Toko A', 'Toko C', 'Toko B'],
            'Nama Barang': ['Semen', 'Paku', 'Cat Tembok', 'Kabel', 'Pipa'],
            'Harga': [50000, 10000, 120000, 15000, 35000],
            'Jumlah': [2, 5, 1, 10, 4]
        }
        df = pd.DataFrame(data)
        st.warning("⚠️ File 'nota.csv' tidak ditemukan. Menggunakan data contoh.")

    # Cleaning Standard
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    def clean_numeric(val):
        if isinstance(val, str):
            val = val.replace(',', '').replace(' ', '').strip()
            try:
                return float(val)
            except:
                return 0
        return float(val) if pd.notnull(val) else 0

    if 'Harga' in df.columns:
        df['Harga'] = df['Harga'].apply(clean_numeric)
    if 'Jumlah' in df.columns:
        df['Jumlah'] = df['Jumlah'].apply(clean_numeric)
    
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    
    if {'Nama Barang', 'Toko', 'Harga'}.issubset(df.columns):
        df = df.dropna(subset=['Nama Barang', 'Toko', 'Harga'])
        df = df[df['Harga'] > 0]
    
    return df

# --- 3. MACHINE LEARNING ENGINE ---
@st.cache_resource
def train_model(df):
    if len(df) < 5: return None, None, None

    df_train = df.copy()
    X_text = df_train['Nama Barang'].astype(str)
    X_toko = df_train['Toko'].astype(str)
    y = df_train['Harga']

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=200) 
    X_text_vec = vectorizer.fit_transform(X_text).toarray()

    # Label Encoder
    le_toko = LabelEncoder()
    all_shops = list(X_toko.unique()) + ['Lainnya']
    le_toko.fit(all_shops)
    X_toko_enc = le_toko.transform(X_toko)

    # Gabungkan
    X_final = pd.DataFrame(X_text_vec)
    X_final['Toko_Code'] = X_toko_enc
    X_final.columns = X_final.columns.astype(str) 

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_final, y)

    return model, vectorizer, le_toko

# --- 4. MAIN APPLICATION ---
def main():
    # --- SIDEBAR ---
    st.sidebar.title("🎛️ Filter Data")
    df = load_data()

    if df.empty:
        st.stop()

    if 'Toko' in df.columns:
        all_shops = sorted(df['Toko'].unique().astype(str))
        selected_shops = st.sidebar.multiselect("Pilih Toko", all_shops, default=all_shops)
        if selected_shops:
            df_filtered = df[df['Toko'].isin(selected_shops)]
        else:
            df_filtered = df
    else:
        df_filtered = df

    st.sidebar.markdown("---")
    st.sidebar.info(f"Total Data: {len(df)} Transaksi")
    
    # --- MAIN CONTENT ---
    st.title("📊 Dashboard Analisa Harga & Belanja")
    st.markdown("Monitor pengeluaran proyek dan cek prediksi harga wajar menggunakan AI.")

    tab1, tab2, tab3 = st.tabs(["📈 Dashboard Eksekutif", "🤖 Cek Harga Wajar (AI)", "📋 Data Mentah"])

    # === TAB 1: DASHBOARD ===
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        total_spend = df_filtered['Jumlah'].sum() if 'Jumlah' in df_filtered.columns else 0
        avg_price = df_filtered['Harga'].mean() if 'Harga' in df_filtered.columns else 0
        top_shop = df_filtered['Toko'].mode()[0] if 'Toko' in df_filtered.columns and not df_filtered.empty else "-"
        
        col1.metric("Total Pengeluaran", f"Rp {total_spend:,.0f}".replace(',', '.'))
        col2.metric("Rata-rata Harga", f"Rp {avg_price:,.0f}".replace(',', '.'))
        col3.metric("Toko Terlaris", str(top_shop))
        col4.metric("Jumlah Transaksi", len(df_filtered))

        st.markdown("---")

        # Charts
        # PENTING: Saya menghapus `template='plotly_dark'` agar chart menyesuaikan tema otomatis.
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tren Pengeluaran Harian")
            if 'Tanggal' in df_filtered.columns:
                daily = df_filtered.groupby('Tanggal')['Jumlah'].sum().reset_index()
                fig_line = px.line(daily, x='Tanggal', y='Jumlah', markers=True, 
                                   line_shape='spline')
                fig_line.update_traces(line_color='#00CC96')
                st.plotly_chart(fig_line, use_container_width=True)
        
        with c2:
            st.subheader("Top 5 Toko")
            if 'Toko' in df_filtered.columns:
                shop_stats = df_filtered.groupby('Toko')['Jumlah'].sum().reset_index()
                shop_stats = shop_stats.sort_values('Jumlah', ascending=False).head(5)
                fig_bar = px.bar(shop_stats, x='Jumlah', y='Toko', orientation='h', 
                                 text_auto='.2s', color='Jumlah', color_continuous_scale='Viridis')
                st.plotly_chart(fig_bar, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Top 5 Barang")
            if 'Nama Barang' in df_filtered.columns:
                item_counts = df_filtered['Nama Barang'].value_counts().head(5).reset_index()
                item_counts.columns = ['Nama Barang', 'Frekuensi']
                fig_pie = px.pie(item_counts, names='Nama Barang', values='Frekuensi', hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True)

        with c4:
            st.subheader("Distribusi Harga")
            if 'Harga' in df_filtered.columns:
                fig_hist = px.histogram(df_filtered, x='Harga', nbins=20, 
                                        color_discrete_sequence=['#AB63FA'])
                st.plotly_chart(fig_hist, use_container_width=True)

    # === TAB 2: AI PREDICTOR ===
    with tab2:
        st.header("🤖 Kalkulator Prediksi Harga (AI)")
        st.write("Masukkan nama barang untuk melihat estimasi harga wajar.")
        
        model, vectorizer, le_toko = train_model(df)

        if model:
            with st.container(): # Hapus styling manual background
                col_in1, col_in2, col_in3 = st.columns([2, 1, 1])
                
                with col_in1:
                    input_item = st.text_input("Nama Barang", placeholder="Contoh: Semen, Paku...")
                with col_in2:
                    shop_options = sorted(list(le_toko.classes_))
                    input_shop = st.selectbox("Toko", shop_options)
                with col_in3:
                    input_qty = st.number_input("Jumlah", min_value=1, value=1)

                if st.button("🔍 Cek Estimasi Harga", type="primary"):
                    if input_item:
                        try:
                            vec_input = vectorizer.transform([input_item]).toarray()
                            shop_input = le_toko.transform([input_shop])
                            
                            pred_df = pd.DataFrame(vec_input)
                            pred_df['Toko_Code'] = shop_input
                            pred_df.columns = pred_df.columns.astype(str)
                            
                            price_est = model.predict(pred_df)[0]
                            total_est = price_est * input_qty

                            st.success("Analisis Selesai!")
                            m1, m2 = st.columns(2)
                            m1.metric("Estimasi Harga Satuan", f"Rp {price_est:,.0f}".replace(',', '.'))
                            m2.metric(f"Total Estimasi ({input_qty} item)", f"Rp {total_est:,.0f}".replace(',', '.'))
                            
                            st.info(f"💡 Prediksi berdasarkan pola nama barang mirip '{input_item}' di '{input_shop}'.")
                        except Exception as e:
                            st.error(f"Gagal memprediksi: {e}")
                    else:
                        st.warning("Mohon isi nama barang.")
        else:
            st.warning("Data belum cukup untuk melatih AI (Minimal 5 data unik).")

    # === TAB 3: DATAVIEW ===
    with tab3:
        st.subheader("📋 Database Nota")
        search_term = st.text_input("Cari data di tabel:", placeholder="Ketik nama barang...", key="search_db")
        
        if search_term:
            df_display = df_filtered[df_filtered['Nama Barang'].str.contains(search_term, case=False, na=False)]
        else:
            df_display = df_filtered

        st.dataframe(df_display, use_container_width=True)
        
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data CSV",
            data=csv,
            file_name='data_nota_filtered.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()