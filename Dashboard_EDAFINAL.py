import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,chi2_contingency

# Cara menjalankan:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\activate
# python -m streamlit run Dashboard_EDAFINAL.py

# ======================
# Konfigurasi Dasar App
# ======================
st.set_page_config(
    page_title="Dashboard Analisis OSADA",
    page_icon="📊",
    layout="wide"
)

# ======================
# Custom CSS Styling Adaptif (Sidebar tetap biru)
# ======================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0077B6 0%, #00B4D8 100%);
    color: white;
}
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 500;
}
.stApp {
    background-color: #F8FCFF;
}
h1, h2, h3 {
    color: #006D77;
}
.stPlotlyChart {
    border-radius: 16px;
    background-color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0, 100, 120, 0.1);
    padding: 20px;
    margin: 10px 0 25px 0;
}

/* ========== MODE ADAPTIF TEKS (Sidebar tetap biru) ========== */

/* Light mode */
@media (prefers-color-scheme: light) {
    html, body, [data-testid="stAppViewContainer"] {
        color: #1E1E1E !important;  /* teks utama gelap */
        background-color: #F8FCFF !important;
    }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    html, body, [data-testid="stAppViewContainer"] {
        color: #EAEAEA !important;  /* teks jadi putih lembut */
        background-color: #121212 !important;  /* latar konten gelap */
    }
    .stPlotlyChart {
        background-color: #1E1E1E !important;  /* chart card gelap */
        box-shadow: 0 0 10px rgba(255,255,255,0.05);
    }
    h1, h2, h3 {
        color: #B0E0FF !important;  /* judul biru muda elegan */
    }
}
</style>
""", unsafe_allow_html=True)


# ======================
# Header & Pembuka
# ======================
st.title("📊 Analisis Pengkaderan: OSADA terhadap pengembangan diri Mahasiswa Sains Data")
st.markdown("---")
st.subheader(" Disusun oleh Kelompok robloxmania - Mata Kuliah Analisis Data Eksploratif\n" "Sulaiman Abhinaya Praditya (24083010041) ; Evelyna Kamila (24083010097) ; Nabilla Roza Meyrina Yolanda (24083010102)")
st.markdown("---")

# ======================
# Load Data
# ======================
# ======================
@st.cache_data
def load_data():
    try:
        df_num = pd.read_csv("data_numerik.csv")
        df_cat = pd.read_csv("data_kategorikal.csv")
        return df_num, df_cat
    except FileNotFoundError:
        return None, None


df_num, df_cat = load_data()
if df_num is None or df_cat is None:
    st.error("❌ Pastikan file data tersedia di direktori kerja.")
    st.stop()


if "NPM" in df_cat.columns:
    df_cat["angkatan"] = df_cat["NPM"].astype(str).str[:2]

# ==============================
# 🎯 Fungsi Interpretasi Berdasarkan Bentuk Grafik
# ==============================

# --- 1. Interpretasi Dominasi Proporsi (untuk Pie / Bar Chart) ---
def interpret_from_shape(series, x_label):
    series = series.dropna()
    total = series.sum()
    proportions = (series / total * 100).sort_values(ascending=False)
    top_label = proportions.index[0]
    top_value = proportions.iloc[0]
    second_value = proportions.iloc[1] if len(proportions) > 1 else 0

    if top_value - second_value > 20:
        return f"Kategori **'{top_label}'** mendominasi pada variabel *{x_label}* dengan proporsi sekitar {top_value:.1f}%."
    elif top_value > 50:
        return f"Sebagian besar responden memilih **'{top_label}'** pada variabel *{x_label}*, menunjukkan kecenderungan kuat."
    elif top_value - second_value < 10:
        return f"Tidak ada dominasi yang jelas pada variabel *{x_label}*; distribusi antar kategori relatif seimbang."
    else:
        return f"Ada kecenderungan ke arah kategori **'{top_label}'**, meskipun selisih dengan kategori lain tidak terlalu besar."

# --- 2. Interpretasi Pola Tren (untuk data ordinal seperti skala 1–5) ---
def interpret_trend(series, x_label):
    try:
        values = pd.to_numeric(series, errors='coerce').dropna()
        if len(values) < 3:
            return ""
        s = skew(values)
        if s < -0.5:
            return f"Pola distribusi menunjukkan kecenderungan ke arah nilai tinggi pada *{x_label}* (mayoritas merasa sulit)."
        elif s > 0.5:
            return f"Distribusi cenderung ke arah nilai rendah pada *{x_label}* (mayoritas merasa mudah)."
        else:
            return f"Distribusi relatif seimbang di *{x_label}*, tanpa dominasi nilai tertentu."
    except Exception:
        return ""

# --- 3. Cramér’s V (untuk kekuatan hubungan antar variabel kategorikal) ---
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def interpret_relation(df, x_col, y_col):
    ct = pd.crosstab(df[x_col], df[y_col])
    strength = cramers_v(ct)
    if strength > 0.5:
        return f"Ada hubungan yang **kuat** antara *{x_col}* dan *{y_col}*."
    elif strength > 0.3:
        return f"Ada hubungan yang **cukup kuat** antara *{x_col}* dan *{y_col}*."
    elif strength > 0.1:
        return f"Ada hubungan yang **lemah** antara *{x_col}* dan *{y_col}*."
    else:
        return f"Tidak terdapat hubungan yang berarti antara *{x_col}* dan *{y_col}*."

# --- 4. Fungsi gabungan untuk insight otomatis ---
def generate_shape_insight(df, x_col, y_col=None):
    """Menghasilkan interpretasi otomatis berdasarkan bentuk grafik."""
    insight_parts = []

    # Insight distribusi tunggal (pie/bar)
    if x_col in df.columns:
        series = df[x_col].value_counts()
        insight_parts.append(interpret_from_shape(series, x_col))
        trend_text = interpret_trend(df[x_col], x_col)
        if trend_text:
            insight_parts.append(trend_text)

    # Insight hubungan antar variabel
    if y_col and y_col in df.columns:
        relation_text = interpret_relation(df, x_col, y_col)
        insight_parts.append(relation_text)

    return " ".join(insight_parts)


# ======================
# Sidebar Navigation
# ======================
with st.sidebar.expander("📊 Menu Navigasi", expanded=True):
    menu = st.radio(
        "Pilih Halaman:",
        ["🚀 Overview Data", "📈 Visualisasi & Hasil Analisis", "🔗 Hubungan Antar Variabel", "🧩 Kesimpulan"]
    )

if menu == "📈 Visualisasi & Hasil Analisis":
    with st.sidebar.expander("📈 Pilih Submenu Analisis", expanded=True):
        vis_choice = st.radio(
            "Fokus Analisis:",
            [
                "📊 Dampak OSADA terhadap Kedisiplinan",
                "🤝 Kegiatan yang Paling Membantu Pengembangan Diri",
                "🔥 Keaktifan setelah Mengikuti OSADA"
            ]
        )
else:
    vis_choice = None

if menu == "🔗 Hubungan Antar Variabel":
    with st.sidebar.expander("🔗 Pilih Jenis Hubungan", expanded=True):
        hub_choice = st.radio(
            "Fokus Hubungan:",
            [
                "🔗 Hubungan antar Variabel Kategorikal",
                "🔗 Hubungan antar Variabel Numerik & Kategorikal"
            ]
        )
else:
    hub_choice = None

if menu == "🧩 Kesimpulan":
    with st.sidebar.expander("🧩 Pilih Bagian Kesimpulan", expanded=True):
        kesimpulan_choice = st.radio(
            "Bagian Kesimpulan:",
            ["📋 Ringkasan Temuan", "💡 Rekomendasi", "🎯 Implikasi"]
        )
else:
    kesimpulan_choice = None

st.sidebar.markdown("---")
st.sidebar.markdown("Gunakan menu di atas untuk navigasi antar halaman dashboard.")
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align:justify;'>Dibuat oleh: <b>Tim Analisis OSADA - © 2025 Kelompok robloxmania 📊</b></div>", unsafe_allow_html=True)

# ======================
# Fungsi bantu
# ======================
def tampilkan_grafik_dengan_interpretasi(fig, text, key):

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(fig, use_container_width=True, key=key)
    with col2:
        st.markdown(f"<div style='text-align:justify;line-height:1.6;'>{text}</div>", unsafe_allow_html=True)
    st.markdown("---")


# ======================
# Konten Halaman
# ======================
if menu == "🚀 Overview Data":
    st.title("🎯 Dashboard Analisis Dampak OSADA")
    st.markdown("""
    ### Selamat Datang di Dashboard Analisis OSADA!
    
    OSADA (Orientasi Sains Data I) adalah kegiatan pengenalan kehidupan kampus bagi mahasiswa baru yang bertujuan memberikan informasi seputar sistem perkuliahan, dosen, organisasi mahasiswa, serta nilai-nilai dasar program studi. Melalui OSADA, mahasiswa baru diharapkan siap menjalani perkuliahan dan aktif berkontribusi di lingkungan kampus.

    ### Insight Utama:
    
    Berdasarkan analisis data responden, OSADA telah membuktikan dampak positif yang signifikan terhadap pengembangan diri mahasiswa. **85,2% mahasiswa** melaporkan peningkatan kedisiplinan setelah mengikuti program ini, sementara **68,2% merasa lebih aktif** dalam berbagai kegiatan akademik maupun non-akademik di kampus.

    Yang menarik, kegiatan berbasis kolaborasi seperti **kerja kelompok dan studi kasus** terbukti paling efektif dalam mendukung pengembangan diri mahasiswa. Analisis lebih lanjut menunjukkan bahwa mahasiswa yang menghabiskan **1-5 jam per minggu** untuk OSADA mengalami peningkatan kedisiplinan paling optimal.

    Temuan lainnya mengungkap bahwa pengalaman **presentasi selama OSADA** berperan sebagai katalisator yang mendorong keaktifan mahasiswa di lingkungan kampus. Meskipun menghadapi tantangan seperti pengurangan jam tidur, motivasi mahasiswa untuk berorganisasi tetap tinggi, menunjukkan dedikasi dan komitmen yang kuat.

    Terakhir pada segi sosial, mahasiswa yang berhasil menjalin **lebih dari 20 teman baru** selama OSADA cenderung lebih aktif berpartisipasi dalam kegiatan kampus. Hal ini menunjukkan bahwa jaringan sosial yang terbentuk selama OSADA berkontribusi pada peningkatan keaktifan mahasiswa.

    Secara keseluruhan, OSADA berhasil menciptakan transformasi melalui pembentukan kebiasaan disiplin, peningkatan kepercayaan diri, dan penguatan komitmen mahasiswa—menjadikannya fondasi yang kokoh untuk kesuksesan akademik dan pengembangan diri selama masa studi.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Responden Merasa Lebih Disiplin", "85%")
    with col2:
        st.metric("Peningkatan Keaktifan Akademik & Non-Akademik", "68%")

elif menu == "📈 Visualisasi & Hasil Analisis":
    if vis_choice == "📊 Dampak OSADA terhadap Kedisiplinan":
        color_discrete_map = {
            'Sangat Tidak Membantu': '#E74C3C',
            'Tidak Membantu': '#FADBD8',
            'Membantu': '#6FAED9',
            'Sangat Membantu': '#1F4E79'
        }
        st.header("📊 Dampak OSADA terhadap Kedisiplinan")
        col_name = "5. Sejauh mana OSADA membantu Anda dalam meningkatkan kedisiplinan?"
        if col_name in df_cat.columns:
            data = df_cat[col_name].value_counts().reset_index()
            data.columns = ['Kategori', 'Jumlah']
            fig = px.pie(data, names='Kategori', values='Jumlah', color='Kategori', title="Distribusi Persepsi Kedisiplinan Mahasiswa", color_discrete_map=color_discrete_map)
            tampilkan_grafik_dengan_interpretasi(fig, "Terlihat dari grafik lingkaran di sebelah, sebagian besar responden menilai OSADA meningkatkan kedisiplinan mereka. Sebanyak 85,7% responden (total jawaban **sangat membantu** dan **membantu**) merasa lebih disiplin setelah mengikuti OSADA. Ini membuktikan bahwa OSADA membawa dampak positif terhadap kedisiplinan mahasiswa.", key="pie_kedisiplinan")

            if 'angkatan' in df_cat.columns:
                freq = df_cat.groupby([col_name, 'angkatan']).size().reset_index(name='jumlah')
                fig_sun = px.sunburst(freq, path=[col_name, 'angkatan'], values='jumlah', color=col_name, color_discrete_map=color_discrete_map, title="Kedisiplinan Berdasarkan Angkatan")
                fig_sun.update_traces(textinfo="label+percent entry")
                tampilkan_grafik_dengan_interpretasi(fig_sun, "Terlihat dari grafik sunburst di sebelah, dari total 51% jawaban **membantu** angkatan 24 merasa OSADA meningkatkan kedisiplinan mereka dengan persentase 17% diikuti angkatan 22 dengan 13%. Disisi lain jawaban **sangat membantu**, menunjukkan angkatan 23 dengan total 12% dan angkatan 24 dengan total 10%. Hal ini memnunjukkan peningkatan kedisiplinan lebih tinggi terhadap mahasiswa baru yang kemungkinan didukung dengan program OSADA yang baik.", key="sunburst_kedisiplinan")

    elif vis_choice == "🤝 Kegiatan yang Paling Membantu Pengembangan Diri":
        color_discrete_map2 = {
            'Study Case materi: Etika dan Moral dalam Kehidupan Mahasiswa': '#6FAED9',
            'Kerja Kelompok terkait Penugasan OSADA': '#1F4E79',
            'Penjelasan Materi di kelas': '#E74C3C',
            'Wawancara HIMASADA': '#FADBD8'
        }
        color_discrete_map2_short = {
            'Study Case': '#6FAED9',
            'Kerja Kelompok OSADA': '#1F4E79',
            'Materi di Kelas': '#E74C3C',
            'Wawancara': '#FADBD8'
        }
        st.header("🤝 Kegiatan yang Paling Membantu Pengembangan Diri")
        col_name = "2. Jenis kegiatan apa yang paling membantu dalam pengembangan diri Anda selama kegiatan OSADA?"
        if col_name in df_cat.columns:
            data = df_cat[col_name].value_counts().reset_index()
            data.columns = ['Kegiatan', 'Jumlah']
            fig = px.pie(data, names='Kegiatan', values='Jumlah', color='Kegiatan', title="Jenis Kegiatan OSADA yang Paling Membantu Pengembangan Diri", color_discrete_map=color_discrete_map2)
            tampilkan_grafik_dengan_interpretasi(fig, "Terlihat dari grafik lingkaran di sebelah, kegiatan Kerja Kelompok terkait Penugasan OSADA merupakan jenis kegiatan yang paling banyak dipilih responden dengan persentase 38%, diikuti oleh Study Case materi: Etika dan Moral dalam Kehidupan Mahasiswa sebesar 32%. Hal ini menunjukkan bahwa pendekatan menggunakan penugasan kolaborasi kelompok dan pendekatan melalui studi kasus dinilai paling efektif dalam pengembangan diri mahasiswa selama mengikuti OSADA.", key="pie_pengembangan")

            if 'angkatan' in df_cat.columns:
                freq = df_cat.groupby([col_name, 'angkatan']).size().reset_index(name='jumlah')
                freq[col_name] = freq[col_name].replace({
                    'Study Case materi: Etika dan Moral dalam Kehidupan Mahasiswa': 'Study Case',
                    'Kerja Kelompok terkait Penugasan OSADA': 'Kerja Kelompok OSADA',
                    'Penjelasan Materi di kelas': 'Materi di Kelas',
                    'Wawancara HIMASADA': 'Wawancara',
                })
                fig_sun = px.sunburst(freq, path=[col_name, 'angkatan'], values='jumlah', color=col_name, color_discrete_map=color_discrete_map2_short, title="Kegiatan Pengembangan Diri Berdasarkan Angkatan (Disingkat)")
                fig_sun.update_traces(textinfo="label+percent entry")
                tampilkan_grafik_dengan_interpretasi(fig_sun, "Terlihat dari grafik sunburst di sebelah, angkatan 24 mendominasi partisipasi dalam kegiatan Kerja Kelompok dengan kontribusi 13% dari total responden, diikuti oleh angkatan 23 dan 22 dengan persentase 10%. Distribusi ini mengindikasikan bahwa mahasiswa dari berbagai angkatan memiliki preferensi yang berbeda terhadap jenis kegiatan, namun secara keseluruhan kegiatan kolaboratif tetap menjadi pilihan utama.", key="sunburst_pengembangan")

    elif vis_choice == "🔥 Keaktifan setelah Mengikuti OSADA":
        color_discrete_map3 = {
            'Sangat Tidak Aktif': '#E74C3C',
            'Tidak Aktif': '#FADBD8',
            'Aktif': '#6FAED9',
            'Sangat Aktif': '#1F4E79'
        }
        st.header("🔥 Keaktifan setelah Mengikuti OSADA")
        col_name = "9.  Apakah setelah mengikuti pengkaderan OSADA Anda merasa lebih aktif dalam kegiatan akademik maupun non-akademik di kampus?"
        if col_name in df_cat.columns:
            data = df_cat[col_name].value_counts().reset_index()
            data.columns = ['Status', 'Jumlah']
            fig = px.pie(data, names='Status', values='Jumlah', color='Status', title="Persepsi Keaktifan Setelah Mengikuti OSADA", color_discrete_map=color_discrete_map3)
            tampilkan_grafik_dengan_interpretasi(fig, "Terlihat dari grafik lingkaran di sebelah, sebanyak 68,2% responden menyatakan merasa aktif dan sangat aktif dalam kegiatan akademik maupun non-akademik setelah mengikuti OSADA. Hanya 22% yang merasa tidak mengalami perubahan signifikan. Data ini membuktikan bahwa OSADA berhasil memotivasi mahasiswa untuk lebih berpartisipasi dalam berbagai kegiatan kampus.", key="pie_keaktifan")

            if 'angkatan' in df_cat.columns:
                freq = df_cat.groupby([col_name, 'angkatan']).size().reset_index(name='jumlah')
                fig_sun = px.sunburst(freq, path=[col_name, 'angkatan'], values='jumlah', color=col_name, color_discrete_map=color_discrete_map3, title="Keaktifan Setelah OSADA Berdasarkan Angkatan")
                fig_sun.update_traces(textinfo="label+percent entry")
                tampilkan_grafik_dengan_interpretasi(fig_sun, "Terlihat dari grafik sunburst di sebelah, angkatan 24 menunjukkan tingkat keaktifan tertinggi pasca OSADA dengan kontribusi 18% dari total responden yang merasa aktif, diikuti angkatan 22 sebesar 14%. Yang menarik, mahasiswa angkatan 23 justru lebih banyak menyatakan merasa sangat aktif dengan persentase 5% diikuti angkatan 22 dengan persentase 4%, mengindikasikan bahwa dampak positif OSADA terhadap keaktifan organisasi dapat bertahan hingga tahun-tahun berikutnya.", key="sunburst_keaktifan")



# ---------- Numerik ----------
elif menu == "🔗 Hubungan Antar Variabel":
    # ---------- Kategorikal ----------
    if hub_choice == "🔗 Hubungan antar Variabel Kategorikal":
        st.header("🔗 Hubungan antar Variabel Kategorikal")
        st.info("Analisis distribusi silang antar dua variabel kategori menggunakan stacked bar chart dan interpretasi otomatis.")

        # --- Daftar kolom kategorikal yang tersedia ---
        categorical_columns = [
            "1. Dari skala 1–4, seberapa sulit penugasan OSADA menurut Anda?",
            "2. Jenis kegiatan apa yang paling membantu dalam pengembangan diri Anda selama kegiatan OSADA?",
            "5. Sejauh mana OSADA membantu Anda dalam meningkatkan kedisiplinan?",
            "9.  Apakah setelah mengikuti pengkaderan OSADA Anda merasa lebih aktif dalam kegiatan akademik maupun non-akademik di kampus?",
            "10.  Apakah OSADA memberikan motivasi tambahan bagi Anda untuk aktif dalam organisasi lain di kampus?"
        ]

        # --- Pilih variabel X dan Y untuk analisis ---
        st.subheader("Pilih Variabel untuk Crosstab")
        x_col = st.selectbox("Variabel X (sebagai dasar bar chart):", categorical_columns, index=0)
        y_col = st.selectbox("Variabel Y (pembeda warna di bar chart):", categorical_columns, index=2)

        # --- Analisis Crosstab ---
        if x_col in df_cat.columns and y_col in df_cat.columns:
            crosstab = pd.crosstab(df_cat[x_col], df_cat[y_col], normalize='index') * 100
            st.write("**Tabel Crosstab (%):**")
            st.dataframe(crosstab.style.format("{:.1f}%"))

            # --- Visualisasi Stacked Bar ---
            fig = px.bar(
                df_cat,
                x=x_col,
                color=y_col,
                barmode='stack',
                title=f"Distribusi Gabungan: {x_col} vs {y_col}",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Interpretasi Singkat ---
            st.markdown("### 🧭 Interpretasi Singkat")

            # Hitung rata-rata proporsi tiap kategori Y
            top_category_y = crosstab.mean().idxmax()
            top_value_y = crosstab.mean().max()

            # Temukan baris (kategori X) dengan proporsi tertinggi untuk kategori Y tersebut
            top_x_row = crosstab[top_category_y].idxmax()
            top_x_value = crosstab.loc[top_x_row, top_category_y]

            # Tampilkan interpretasi dengan dua arah (X dan Y)
            st.markdown(f"""
            <div style="text-align: justify; line-height: 1.6;">
            Berdasarkan hasil crosstab, variabel <b>"{y_col}"</b> menunjukkan bahwa kategori 
            <b>"{top_category_y}"</b> memiliki proporsi rata-rata tertinggi sebesar 
            <b>{top_value_y:.1f}%</b> di seluruh kelompok <b>"{x_col}"</b>.  
            Menariknya, proporsi tertinggi untuk kategori tersebut ditemukan pada kelompok 
            <b>"{top_x_row}"</b> dengan nilai sebesar <b>{top_x_value:.1f}%</b>.  
            Hal ini mengindikasikan bahwa responden yang berada pada kelompok <b>"{top_x_row}"</b> 
            cenderung lebih banyak memberikan penilaian <b>"{top_category_y}"</b> pada variabel 
            <b>"{y_col}"</b>.  
            Visualisasi stacked bar chart memperkuat temuan ini dengan menunjukkan dominasi warna 
            yang sesuai pada kelompok tersebut.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Variabel yang dipilih tidak ditemukan dalam data kategorikal.")

    # ---------- Numerik ----------
    elif hub_choice == "🔗 Hubungan antar Variabel Numerik & Kategorikal":
        st.header("🔗 Hubungan Antar Variabel Numerik & Kategorikal")
        st.info("Analisis hubungan antara variabel numerik dan kategorikal untuk melihat pengaruh kegiatan OSADA terhadap perkembangan diri mahasiswa.")

        # =====================================================
        # 1️⃣ Waktu OSADA vs Kedisiplinan
        # =====================================================
        st.subheader("⏰ 1. Waktu OSADA vs Kedisiplinan")
        
        waktu_col = "3. Berapa rata-rata waktu yang Anda habiskan per minggu untuk mengerjakan tugas OSADA?"
        kedisiplinan_col = "5. Sejauh mana OSADA membantu Anda dalam meningkatkan kedisiplinan?"

        if waktu_col in df_num.columns and kedisiplinan_col in df_cat.columns:
            # Konversi ke numerik
            df_num['waktu_numeric'] = pd.to_numeric(df_num[waktu_col], errors='coerce')

            # Buat dataframe untuk plot
            plot_df = pd.DataFrame({
                'Waktu': df_num['waktu_numeric'],
                'Kedisiplinan': df_cat[kedisiplinan_col]
            }).dropna()
            
            if not plot_df.empty:
                # Buat kelompok waktu
                plot_df['Kelompok_Waktu'] = pd.cut(
                    plot_df['Waktu'],
                    bins=[0, 5, 10, 15, 100],
                    labels=['1–5 jam', '6–10 jam', '11–15 jam', '>15 jam']
                )
                
                # Buat plot
                fig1 = px.bar(
                    plot_df,
                    x='Kelompok_Waktu',
                    color='Kedisiplinan',
                    title="Hubungan Waktu Pengerjaan Tugas dengan Kedisiplinan",
                    barmode='stack',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Interpretasi
                st.markdown("""
                **Interpretasi:**  
                Semakin sedikit waktu yang dihabiskan untuk OSADA, semakin tinggi tingkat kedisiplinan yang dilaporkan.  
                Mahasiswa yang menghabiskan 1-5 jam per minggu menunjukkan peningkatan kedisiplinan paling signifikan.  
                Sekitar 85% responden merasa OSADA membantu meningkatkan kedisiplinan mereka, dengan hanya 15% yang merasa kurang terbantu.
                Hal ini menunjukkan bahwa efisiensi dalam mengelola waktu OSADA berkontribusi positif terhadap pembentukan kebiasaan disiplin di kalangan mahasiswa.
                """)
            else:
                st.warning("Tidak ada data valid untuk analisis waktu vs kedisiplinan.")
        else:
            st.error("Kolom data tidak ditemukan.")

        st.divider()

        # =====================================================
        # 2️⃣ Presentasi vs Keaktifan
        # =====================================================
        st.subheader("🎤 2. Presentasi vs Keaktifan")
        
        presentasi_col = "6. Berapa jumlah presentasi atau kesempatan berbicara di depan umum yang Anda lakukan selama OSADA?"
        keaktifan_col = "9.  Apakah setelah mengikuti pengkaderan OSADA Anda merasa lebih aktif dalam kegiatan akademik maupun non-akademik di kampus?"

        if presentasi_col in df_num.columns and keaktifan_col in df_cat.columns:
            # Konversi ke numerik
            df_num['presentasi_numeric'] = pd.to_numeric(df_num[presentasi_col], errors='coerce')
            
            # Buat dataframe untuk plot
            plot_df2 = pd.DataFrame({
                'Presentasi': df_num['presentasi_numeric'],
                'Keaktifan': df_cat[keaktifan_col]
            }).dropna()
            
            if not plot_df2.empty:
                # Buat kelompok presentasi
                plot_df2['Kelompok_Presentasi'] = pd.cut(
                    plot_df2['Presentasi'],
                    bins=[-1, 0, 2, 4, 100],
                    labels=['Tidak Pernah', '1–2 kali', '3–4 kali', '≥5 kali']
                )
                
                # Buat plot
                fig2 = px.bar(
                    plot_df2,
                    x='Kelompok_Presentasi',
                    color='Keaktifan',
                    title="Hubungan Frekuensi Presentasi dengan Keaktifan Pasca OSADA",
                    barmode='stack',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Interpretasi
                st.markdown("""
                **Interpretasi:**  
                Frekuensi presentasi selama OSADA berkorelasi positif dengan tingkat keaktifan di kampus.  
                Mahasiswa yang presentasi 1–2 kali sudah menunjukkan peningkatan keaktifan, sementara yang tidak presentasi sama sekali masih merasa aktif melihat hampir 70% memilih **aktif**.  
                Walaupun pengalaman berbicara di depan umum dapat membangun kepercayaan diri untuk berpartisipasi dalam kegiatan kampus, nyatanya masih ada yang tidak pernah melakukan presentasi masih merasa aktif.
                Menandakan ada kemungkinan bahwa kesempatan presentasi tidak merata saat OSADA.
                """)
            else:
                st.warning("Tidak ada data valid untuk analisis presentasi vs keaktifan.")
        else:
            st.error("Kolom data tidak ditemukan.")

        st.divider()

        # =====================================================
        # 3️⃣ Tidur vs Motivasi
        # =====================================================
        st.subheader("😴 3. Tidur vs Motivasi")
        
        tidur_col = "4. Berapa total jam tidur Anda yang berkurang per minggu selama mengikuti OSADA?"
        motivasi_col = "10.  Apakah OSADA memberikan motivasi tambahan bagi Anda untuk aktif dalam organisasi lain di kampus?"

        if tidur_col in df_num.columns and motivasi_col in df_cat.columns:
            # Konversi ke numerik
            df_num['tidur_numeric'] = pd.to_numeric(df_num[tidur_col], errors='coerce')

            # Buat dataframe untuk plot
            plot_df3 = pd.DataFrame({
                'Tidur': df_num['tidur_numeric'],
                'Motivasi': df_cat[motivasi_col]
            }).dropna()
            
            if not plot_df3.empty:
                # Buat kelompok tidur
                plot_df3['Kelompok_Tidur'] = pd.cut(
                    plot_df3['Tidur'],
                    bins=[0, 2, 5, 8, 100],
                    labels=['0–2 jam', '3–5 jam', '6–8 jam', '>8 jam']
                )
                
                # Buat plot
                fig3 = px.bar(
                    plot_df3,
                    x='Kelompok_Tidur',
                    color='Motivasi',
                    title="Hubungan Pengurangan Jam Tidur dengan Motivasi Organisasi",
                    barmode='stack',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Interpretasi
                st.markdown("""
                **Interpretasi:**  
                Meskipun mengurangi jam tidur, mayoritas mahasiswa (70–80%) tetap termotivasi untuk aktif berorganisasi.  
                Kelompok dengan pengurangan tidur 3–5 jam justru melaporkan motivasi tertinggi.
                Ini menandakan bahwa semakin sedikit jam tidur yang dikorbankan, semakin besar motivasi untuk berpartisipasi dalam organisasi kampus.
                Namun, pengurangan tidur yang berlebihan (6-8 jam) tampaknya berdampak negatif pada motivasi, dengan proporsi motivasi yang lebih rendah dibandingkan kelompok lainnya.
                """)
            else:
                st.warning("Tidak ada data valid untuk analisis tidur vs motivasi.")
        else:
            st.error("Kolom data tidak ditemukan.")

        st.divider()

        # =====================================================
        # 4️⃣ Teman Baru vs Keaktifan Pasca OSADA
        # =====================================================
        st.subheader("🤝 4. Teman Baru vs Keaktifan Pasca OSADA")

        teman_col = "8. Seberapa banyak teman baru yang Anda kenal dari pengkaderan OSADA?"
        keaktifan_col = "9.  Apakah setelah mengikuti pengkaderan OSADA Anda merasa lebih aktif dalam kegiatan akademik maupun non-akademik di kampus?"

        if teman_col in df_num.columns and keaktifan_col in df_cat.columns:
            # Pastikan kolom numerik valid
            df_num['teman_numeric'] = pd.to_numeric(df_num[teman_col], errors='coerce')
            df_mix = pd.DataFrame({
                'Teman': df_num['teman_numeric'],
                'Keaktifan': df_cat[keaktifan_col]
            }).dropna()

            if not df_mix.empty:
                # Kelompokkan jumlah teman baru agar terbaca jelas
                bins = [0, 5, 10, 20, df_mix['Teman'].max()]
                labels = ['1–5 orang', '6–10 orang', '11–20 orang', '>20 orang']
                df_mix['Kelompok_Teman'] = pd.cut(df_mix['Teman'], bins=bins, labels=labels, include_lowest=True)

                # Crosstab proporsi
                crosstab = pd.crosstab(df_mix['Kelompok_Teman'], df_mix['Keaktifan'], normalize='index') * 100

                # 📊 Stacked Bar Chart
                fig_bar = px.bar(
                    df_mix,
                    x='Kelompok_Teman',
                    color='Keaktifan',
                    barmode='stack',
                    title="Hubungan antara Jumlah Teman Baru dan Keaktifan Pasca OSADA",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_bar.update_layout(
                    xaxis_title="Kelompok Jumlah Teman Baru",
                    yaxis_title="Jumlah Responden",
                    legend_title="Tingkat Keaktifan"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # 🧭 Interpretasi
                top_category_y = crosstab.mean().idxmax()
                top_value_y = crosstab.mean().max()
                top_x_row = crosstab[top_category_y].idxmax()
                top_x_value = crosstab.loc[top_x_row, top_category_y]

                st.markdown(f"""
                **Interpretasi:**  
                Jumlah teman baru yang diperoleh selama OSADA berpengaruh kuat terhadap keaktifan pasca program. Responden dengan teman baru lebih dari 20 orang hampir seluruhnya merasa "Aktif" atau "Sangat Aktif". Sebaliknya, yang mendapat sedikit teman baru cenderung kurang aktif. Ini menandakan bahwa semakin sedikit teman baru yang didapat, semakin rendah tingkat keaktifan yang dirasakan. Hal ini menunjukkan bahwa jaringan sosial yang terbentuk selama OSADA menjadi pendorong penting partisipasi dalam kegiatan kampus.
                """)
            else:
                st.warning("Tidak ada data valid untuk analisis jumlah teman vs keaktifan.")
        else:
            st.error("Kolom data untuk jumlah teman baru atau keaktifan tidak ditemukan.")


elif menu == "🧩 Kesimpulan":
    if kesimpulan_choice == "📋 Ringkasan Temuan":
        st.header("📋 Ringkasan Temuan")
        st.markdown("""
        Berdasarkan analisis mendalam terhadap data responden, OSADA telah membuktikan efektivitasnya dalam menciptakan transformasi positif pada mahasiswa. Program ini berhasil meningkatkan kedisiplinan melalui pengelolaan waktu yang optimal, dimana mahasiswa yang menghabiskan 1-5 jam per minggu menunjukkan perkembangan terbaik. 

        Presentasi selama OSADA terbukti menjadi katalisator penting yang mendorong keaktifan mahasiswa di lingkungan kampus. Yang menarik, meskipun menghadapi tantangan seperti pengurangan jam tidur, motivasi untuk berorganisasi justru semakin menguat. Selain itu, perluasan jaringan sosial melalui pertemanan baru selama OSADA berkontribusi signifikan terhadap partisipasi aktif dalam berbagai kegiatan kampus.

        Kegiatan kolaboratif seperti kerja kelompok dan studi kasus dinilai paling efektif dalam mendukung pengembangan diri, sementara tantangan akademik yang seimbang berhasil mempertahankan motivasi belajar mahasiswa.
        """)
    elif kesimpulan_choice == "🎯 Implikasi":
        st.header("🎯 Implikasi")
        st.markdown("""
        Temuan ini menunjukkan bahwa OSADA memiliki potensi untuk menjadi model standar pengembangan diri mahasiswa baru yang dapat direplikasi di berbagai program studi. Membuktikan bahwa OSPEK yang terkadang terlihat tidak bermanfaat, ternyata masih memiliki nilai di dalamnya. Peningkatan kedisiplinan yang dicapai melalui program ini tidak hanya bermanfaat untuk kesuksesan akademik, tetapi juga membentuk kebiasaan yang berguna untuk kehidupan profesional di masa depan.

        Keaktifan organisasi yang tumbuh pasca-OSADA memperkaya pengalaman mahasiswa di luar ruang kuliah, menciptakan lulusan yang lebih seimbang antara hard skills dan soft skills. Transformasi yang terjadi membuktikan bahwa program orientasi yang terstruktur dengan baik dapat menjadi investasi jangka panjang dalam membentuk karakter dan kompetensi mahasiswa.
        """)
    elif kesimpulan_choice == "💡 Rekomendasi":
        st.header("💡 Rekomendasi")
        st.markdown("""
        Untuk mengoptimalkan dampak OSADA ke depannya, disarankan untuk memperbanyak kegiatan berbasis kolaborasi seperti diskusi kelompok dan proyek tim yang telah terbukti efektif. Tingkat kesulitan tugas yang seimbang perlu dipertahankan karena berhasil menciptakan tantangan yang memotivasi tanpa membuat mahasiswa kewalahan.

        Integrasi yang lebih erat dengan kegiatan organisasi kampus lainnya dapat memperkuat dampak keaktifan mahasiswa pasca-OSADA. Selain itu, penyediaan kesempatan presentasi yang lebih banyak akan membantu membangun kepercayaan diri dan kemampuan komunikasi mahasiswa. Pengembangan mekanisme untuk memfasilitasi perluasan jaringan pertemanan juga direkomendasikan untuk mendukung keaktifan berkelanjutan.
        """)





