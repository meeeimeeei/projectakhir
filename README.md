# Analisis Sentimen Publik terhadap DJP

Proyek Data Analytics: klasifikasi sentimen postingan Threads terhadap Direktorat
Jenderal Pajak ke dalam tiga kelas (positif, negatif, netral).

## Struktur

```
proyek_sentimen_djp/
├── data/
│   └── Buku_Kode_dan_Pelabelan_Sentimen_DJP.xlsx
│                                        Buku kode, lembar pengodean, 2 lembar
│                                        uji keandalan, ringkasan, log duplikat
├── notebook/
│   └── analisis_sentimen_djp.ipynb     Notebook analisis lengkap
├── app/
│   ├── preprocessing.py                Modul preprocessing (dipakai bersama)
│   ├── reliabilitas.py                 Krippendorff's alpha + matriks antarpelabel
│   └── app.py                          Aplikasi Streamlit
├── artefak/                            Model terlatih + metadata (dibuat notebook)
├── gambar/                             Grafik (dibuat notebook)
├── laporan/
│   └── Laporan_Analisis_Sentimen_DJP.docx
└── requirements.txt
```

## Urutan menjalankan

**1. Pasang kebutuhan**

```bash
pip install -r requirements.txt
```

**2. Baca buku kode, lalu uji keandalan awal**

Buka `data/Buku_Kode_dan_Pelabelan_Sentimen_DJP.xlsx`. Baca sheet **Buku Kode**
seluruhnya lebih dulu — ini bukan lampiran, melainkan definisi operasional yang
menentukan mutu seluruh hasil.

Kerjakan sheet **Uji Keandalan Awal** (40 unit) sebelum pengodean penuh. Dua
pelabel mengisi kolom masing-masing tanpa saling melihat; bila hanya satu orang,
isi kolom B pada hari berbeda tanpa melihat kolom A. Jalankan bagian 2.5 notebook
untuk menghitung alpha.

- alpha < 0,667 → perbaiki buku kode, ulangi. Jangan mulai pengodean penuh.
- 0,667–0,800 → boleh lanjut, hasil dinyatakan sementara.
- alpha ≥ 0,800 → lanjut.

**3. Pengodean penuh**

Isi kolom **LABEL BARU** pada sheet **Relabeling** (264 baris). Dropdown:
Positif, Negatif, Netral, Tidak Relevan. Kolom **ASPEK** opsional, untuk
pengembangan berbasis aspek nanti.

Setelah itu kerjakan sheet **Uji Keandalan Akhir** (50 unit lain). Nilai alpha
dari lembar inilah yang dilaporkan bersama hasil.

**4. Jalankan notebook**

```bash
jupyter lab
```

Buka `notebook/analisis_sentimen_djp.ipynb`, jalankan seluruh sel dari atas ke
bawah. Notebook otomatis memakai kolom LABEL BARU bila sudah terisi minimal
90 persen, dan jatuh ke Label Lama disertai peringatan bila belum.

Keluaran: `artefak/model_sentimen.joblib`, `artefak/metadata_model.json`, dan
seluruh grafik di `gambar/`.

**5. Jalankan aplikasi**

```bash
streamlit run app/app.py
```

Aplikasi menolak berjalan bila artefak model belum ada, sehingga langkah 4 tidak
dapat terlewat tanpa disadari.

## Landasan metodologis

Prosedur pelabelan mengikuti kaidah analisis isi, bukan sekadar penandaan data:

- **Krippendorff (2019)** — definisi tiga jenis unit (sampling, rekam, konteks);
  keandalan diukur dengan alpha, bukan persentase kesepakatan, dengan ambang
  0,800 dan 0,667.
- **Neuendorf (2017)** — buku kode dan lembar pengodean disusun sebelum
  pengodean; keandalan diuji pada dua titik (awal dan akhir); matriks silang
  antarpelabel dipakai untuk menemukan kategori yang definisinya belum tegas.
- **Liu (2015)** — opini sebagai quintuple (e, a, s, h, t). Entitas ditetapkan
  DJP, aspek GENERAL. Aturan fakta-berimplikasi, sentimen rasional, dan sarkasme
  masuk ke aturan keputusan buku kode.

## Catatan penting

Selama pengodean ulang belum selesai, seluruh angka kinerja yang muncul di notebook,
aplikasi, dan laporan berasal dari label lama yang diketahui tidak konsisten
(terdapat tujuh kelompok teks identik dengan label berbeda). Angka tersebut sah
dipakai sebagai garis dasar pembanding, bukan sebagai hasil akhir.

Setelah relabeling selesai, jalankan ulang notebook, lalu perbarui angka pada
Bab 7 dan Bab 12 laporan.

## Ringkasan angka saat ini (garis dasar, label lama)

| Butir | Nilai |
|---|---|
| Data awal | 289 baris |
| Duplikat dibuang | 25 baris |
| Data unik | 264 baris |
| Data berlabel untuk pemodelan | 263 baris |
| Distribusi | negatif 126, netral 75, positif 62 |
| Model terpilih | Logistic Regression + TF-IDF (1–2 gram) |
| Akurasi (5-fold CV) | 0,567 |
| F1 macro (5-fold CV) | 0,522 |
| Pembanding kelas mayoritas | F1 macro 0,216 |
