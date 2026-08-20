"""
Prototipe aplikasi Analisis Sentimen Publik terhadap DJP.

Menjalankan:
    cd proyek_sentimen_djp
    streamlit run app/app.py

Aplikasi membaca artefak yang dihasilkan notebook:
    artefak/model_sentimen.joblib
    artefak/metadata_model.json
"""

import os
import io
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import preprocess

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_MODEL = os.path.join(BASE, "artefak", "model_sentimen.joblib")
PATH_META = os.path.join(BASE, "artefak", "metadata_model.json")
PATH_HASIL = os.path.join(BASE, "artefak", "hasil_prediksi.csv")
DIR_GAMBAR = os.path.join(BASE, "gambar")

WARNA = {"positif": "#2E7D32", "negatif": "#C62828", "netral": "#616161"}
IKON = {"positif": "▲", "negatif": "▼", "netral": "■"}

st.set_page_config(
    page_title="Analisis Sentimen Publik terhadap DJP",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Pemuatan artefak
# ---------------------------------------------------------------------------
@st.cache_resource
def muat_model():
    if not os.path.exists(PATH_MODEL):
        return None, None
    model = joblib.load(PATH_MODEL)
    meta = {}
    if os.path.exists(PATH_META):
        with open(PATH_META) as f:
            meta = json.load(f)
    return model, meta


@st.cache_data
def muat_hasil():
    if os.path.exists(PATH_HASIL):
        return pd.read_csv(PATH_HASIL)
    return None


model, meta = muat_model()

if model is None:
    st.error(
        "Model belum tersedia. Jalankan notebook "
        "`notebook/analisis_sentimen_djp.ipynb` sampai selesai terlebih dahulu, "
        "supaya file `artefak/model_sentimen.joblib` terbentuk."
    )
    st.stop()


def prediksi(teks):
    """Kembalikan (label, dict probabilitas, teks_hasil_preprocessing)."""
    bersih = preprocess(teks)
    if not bersih.strip():
        return None, None, bersih
    label = model.predict([bersih])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba([bersih])[0]
        proba = dict(zip(model.classes_, p))
    elif hasattr(model.named_steps.get("clf"), "decision_function"):
        d = model.decision_function([bersih])[0]
        e = np.exp(d - np.max(d))
        proba = dict(zip(model.classes_, e / e.sum()))
    return label, proba, bersih


def kata_penentu(bersih, label, n=6):
    """Kata dalam teks yang paling mendorong ke label terpilih."""
    clf = model.named_steps.get("clf")
    vec = model.named_steps.get("tfidf")
    if clf is None or vec is None or not hasattr(clf, "coef_"):
        return []
    idx = list(clf.classes_).index(label)
    koef = clf.coef_[idx]
    nama = vec.get_feature_names_out()
    x = vec.transform([bersih])
    skor = []
    for j in x.nonzero()[1]:
        skor.append((nama[j], float(koef[j] * x[0, j])))
    skor.sort(key=lambda t: -t[1])
    return [s for s in skor[:n] if s[1] > 0]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Informasi Model")
    st.metric("Algoritma", meta.get("model", "-"))
    c1, c2 = st.columns(2)
    c1.metric("F1 Macro", f"{meta.get('f1_macro_cv', 0):.3f}")
    c2.metric("Akurasi", f"{meta.get('akurasi_cv', 0):.3f}")
    st.caption("Diukur dengan stratified 5-fold cross validation.")

    alpha = meta.get("keandalan_alpha")
    st.write("**Keandalan pengodean**")
    if alpha is None:
        st.caption(
            "Krippendorff's alpha belum dihitung. Isi lembar uji keandalan pada "
            "buku kode, lalu jalankan ulang notebook."
        )
    else:
        st.metric("Krippendorff's alpha", f"{alpha:.3f}")
        if alpha >= 0.800:
            st.caption("Di atas 0,800 — pelabelan dapat diandalkan.")
        elif alpha >= 0.667:
            st.caption("Antara 0,667 dan 0,800 — hasil bersifat sementara.")
        else:
            st.caption("Di bawah 0,667 — pelabelan belum layak dipakai.")

    st.divider()
    st.write("**Data latih**")
    st.write(f"Jumlah baris: {meta.get('jumlah_data', '-')}")
    dist = meta.get("distribusi_kelas", {})
    for k in ["positif", "negatif", "netral"]:
        if k in dist:
            st.write(f"- {k.capitalize()}: {dist[k]}")
    st.write(f"Jumlah fitur: {meta.get('jumlah_fitur', '-')}")

    if "BELUM" in str(meta.get("sumber_label", "")):
        st.divider()
        st.warning(
            "Model ini dilatih memakai label lama yang diketahui tidak konsisten. "
            "Angka performa di atas adalah baseline pembanding, bukan hasil akhir. "
            "Latih ulang setelah relabeling selesai."
        )

    st.divider()
    st.caption(
        "Prototipe untuk keperluan proyek. Keluaran model adalah alat penyaring "
        "awal, bukan pengganti pembacaan manusia. Komposisi sentimen di sini "
        "menggambarkan isi dataset, bukan opini publik secara umum, karena "
        "sampel tidak ditarik secara acak."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Analisis Sentimen Publik terhadap DJP")
st.caption(
    "Entitas sentimen: DJP sebagai institusi (kebijakan, layanan, pegawai). "
    "Klasifikasi pada tingkat dokumen, aspek GENERAL."
)

tab1, tab2, tab3 = st.tabs(
    ["Prediksi Teks", "Prediksi Massal", "Gambaran Data Latih"]
)

# ---------------------------------------------------------------------------
# TAB 1
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Klasifikasi satu postingan")

    contoh = {
        "(tulis sendiri)": "",
        "Contoh positif": "Terima kasih DJP, lapor SPT tahunan sekarang sangat mudah dan cepat. Mantap sekali websitenya!",
        "Contoh negatif": "Pajak makin mencekik, tapi hasilnya dikorupsi terus. Percuma bayar",
        "Contoh netral": "Batas waktu penyampaian SPT Tahunan PPh Orang Pribadi adalah tanggal 31 Maret setiap tahunnya",
    }
    pilih = st.selectbox("Muat contoh", list(contoh.keys()))
    teks = st.text_area(
        "Teks postingan",
        value=contoh[pilih],
        height=150,
        placeholder="Tempel teks postingan di sini...",
    )

    if st.button("Analisis", type="primary"):
        if not teks.strip():
            st.warning("Teks masih kosong.")
        else:
            label, proba, bersih = prediksi(teks)
            if label is None:
                st.warning(
                    "Setelah preprocessing tidak ada kata tersisa yang bisa dianalisis. "
                    "Teks kemungkinan hanya berisi emoji, angka, atau tautan."
                )
            else:
                st.markdown(
                    f"<h2 style='color:{WARNA[label]};margin-bottom:0'>"
                    f"{IKON[label]} {label.upper()}</h2>",
                    unsafe_allow_html=True,
                )

                if proba:
                    keyakinan = max(proba.values())
                    st.progress(float(keyakinan))
                    st.caption(f"Tingkat keyakinan model: {keyakinan:.1%}")
                    if keyakinan < 0.5:
                        st.info(
                            "Keyakinan rendah. Postingan seperti ini sebaiknya "
                            "diperiksa manual."
                        )

                    df_p = pd.DataFrame(
                        {"Kelas": list(proba.keys()), "Probabilitas": list(proba.values())}
                    ).sort_values("Probabilitas", ascending=False)
                    st.bar_chart(df_p.set_index("Kelas"), height=200)

                with st.expander("Rincian pemrosesan"):
                    st.write("**Teks setelah preprocessing**")
                    st.code(bersih or "(kosong)")
                    penentu = kata_penentu(bersih, label)
                    if penentu:
                        st.write(f"**Kata yang paling mendorong ke kelas {label}**")
                        st.dataframe(
                            pd.DataFrame(penentu, columns=["Kata", "Kontribusi"]),
                            hide_index=True,
                            width="stretch",
                        )

# ---------------------------------------------------------------------------
# TAB 2
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Klasifikasi banyak postingan sekaligus")
    st.caption(
        "Unggah berkas CSV atau Excel, lalu pilih kolom yang berisi teks postingan."
    )

    berkas = st.file_uploader("Berkas data", type=["csv", "xlsx"])

    if berkas is not None:
        try:
            if berkas.name.lower().endswith(".csv"):
                data = pd.read_csv(berkas)
            else:
                data = pd.read_excel(berkas)
        except Exception as e:
            st.error(f"Berkas gagal dibaca: {e}")
            data = None

        if data is not None and len(data):
            st.write(f"Terbaca {len(data)} baris, {len(data.columns)} kolom.")
            kolom = st.selectbox("Kolom teks", data.columns.tolist())

            if st.button("Jalankan klasifikasi", type="primary"):
                teks_list = data[kolom].astype(str).tolist()
                bersih_list = [preprocess(t) for t in teks_list]

                valid = [i for i, b in enumerate(bersih_list) if b.strip()]
                hasil = pd.Series(["(tidak dapat diproses)"] * len(data))
                keyakinan = pd.Series([np.nan] * len(data))

                if valid:
                    sub = [bersih_list[i] for i in valid]
                    pred = model.predict(sub)
                    hasil.iloc[valid] = pred
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(sub)
                        keyakinan.iloc[valid] = p.max(axis=1)

                keluaran = data.copy()
                keluaran["sentimen"] = hasil.values
                keluaran["keyakinan"] = keyakinan.values.round(3)

                st.success(f"Selesai. {len(valid)} baris berhasil diklasifikasi.")

                ringkas = (
                    keluaran["sentimen"].value_counts().rename_axis("Kelas")
                    .reset_index(name="Jumlah")
                )
                ringkas["Persen"] = (
                    ringkas["Jumlah"] / ringkas["Jumlah"].sum() * 100
                ).round(1)

                k1, k2 = st.columns([1, 2])
                with k1:
                    st.dataframe(ringkas, hide_index=True, width="stretch")
                with k2:
                    st.bar_chart(ringkas.set_index("Kelas")["Jumlah"], height=240)

                ragu = keluaran[keluaran["keyakinan"] < 0.5]
                if len(ragu):
                    st.info(
                        f"{len(ragu)} baris diklasifikasi dengan keyakinan di bawah 50 persen "
                        "dan sebaiknya diperiksa manual."
                    )

                st.dataframe(keluaran.head(50), width="stretch")

                buf = io.BytesIO()
                keluaran.to_csv(buf, index=False)
                st.download_button(
                    "Unduh hasil (CSV)",
                    data=buf.getvalue(),
                    file_name="hasil_sentimen.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------------------
# TAB 3
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Gambaran data latih dan performa model")

    dist = meta.get("distribusi_kelas", {})
    if dist:
        kol = st.columns(len(dist))
        total = sum(dist.values())
        for kolom_, (k, v) in zip(kol, dist.items()):
            kolom_.metric(k.capitalize(), v, f"{v / total:.0%} dari total")

    gambar = [
        ("eda_distribusi.png", "Distribusi kelas dan panjang teks"),
        ("perbandingan_model.png", "Perbandingan kandidat model"),
        ("confusion_matrix.png", "Confusion matrix model terpilih"),
        ("eda_kata_teratas.png", "Kata paling sering muncul per kelas"),
        ("fitur_penting.png", "Kata penentu tiap kelas"),
    ]
    tersedia = [(f, j) for f, j in gambar if os.path.exists(os.path.join(DIR_GAMBAR, f))]

    if not tersedia:
        st.info("Gambar belum tersedia. Jalankan notebook sampai selesai.")
    else:
        for i in range(0, len(tersedia), 2):
            kols = st.columns(2)
            for kolom_, (f, j) in zip(kols, tersedia[i:i + 2]):
                with kolom_:
                    st.image(os.path.join(DIR_GAMBAR, f), caption=j, width="stretch")

    hasil_cv = muat_hasil()
    # if hasil_cv is not None:
   #     st.divider()
  #      st.write("**Kasus salah klasifikasi**")
  #      st.caption(
   #         "Baris yang label sebenarnya berbeda dari prediksi cross validation. "
  #          "Bagian ini dipakai untuk menemukan pola kelemahan model."
  #      )
  #      salah = hasil_cv[hasil_cv["label"] != hasil_cv["prediksi_cv"]]
  #      st.write(f"{len(salah)} dari {len(hasil_cv)} baris salah klasifikasi.")
 #       st.dataframe(
  #          salah[["Isi Thread", "label", "prediksi_cv"]].head(30),
   #         hide_index=True,
     #       width="stretch",
        )
