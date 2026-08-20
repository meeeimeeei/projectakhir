"""
Krippendorff's alpha untuk data nominal.

Dipakai untuk mengukur keandalan pelabelan (reliability of coding) sesuai
Krippendorff (2019, Bab 12). Alpha dipilih, bukan persentase kesepakatan mentah,
karena persentase kesepakatan tidak mengoreksi kesepakatan yang dapat terjadi
secara kebetulan, dan nilainya menyesatkan ketika distribusi kelas timpang.

Standar yang direkomendasikan Krippendorff:
    alpha >= 0,800  -> data dapat diandalkan
    0,667 <= alpha < 0,800 -> hanya untuk kesimpulan sementara (tentatif)
    alpha < 0,667   -> data belum layak dipakai

Alpha juga menangani jumlah pengamat berapa pun dan data yang tidak lengkap,
sehingga tidak masalah bila sebagian unit hanya dilabeli oleh satu pelabel.
"""

from collections import Counter
import numpy as np


def _matriks_nilai(data):
    """
    data: daftar baris, satu baris per pelabel, berisi label tiap unit.
           Nilai None atau string kosong berarti unit tidak dilabeli pelabel itu.
    Mengembalikan daftar unit; setiap unit berupa daftar label yang diberikan.
    """
    n_unit = max(len(baris) for baris in data)
    unit = []
    for i in range(n_unit):
        nilai = []
        for baris in data:
            if i < len(baris):
                v = baris[i]
                if v is not None and str(v).strip() != '' and str(v).lower() != 'nan':
                    nilai.append(str(v).strip())
        unit.append(nilai)
    return unit


def krippendorff_alpha(data):
    """
    Hitung alpha nominal.

    data : daftar berisi daftar. Satu baris per pelabel.
           Contoh dua pelabel, empat unit:
               [['positif', 'negatif', None, 'netral'],
                ['positif', 'netral',  None, 'netral']]

    Mengembalikan float. Nilai 1,0 berarti kesepakatan sempurna; 0,0 berarti
    kesepakatan setara dengan kebetulan; nilai negatif berarti ketidaksepakatan
    sistematis, lebih buruk daripada kebetulan.
    """
    unit = [u for u in _matriks_nilai(data) if len(u) >= 2]
    if not unit:
        raise ValueError('Tidak ada unit yang dilabeli oleh minimal dua pelabel.')

    # n_total = jumlah seluruh nilai yang dapat dipasangkan
    n_total = sum(len(u) for u in unit)

    # Ketidaksepakatan teramati
    Do = 0.0
    for u in unit:
        m = len(u)
        c = Counter(u)
        pasangan_beda = m * (m - 1) - sum(v * (v - 1) for v in c.values())
        Do += pasangan_beda / (m - 1)
    Do /= n_total

    # Ketidaksepakatan yang diharapkan secara kebetulan
    total = Counter()
    for u in unit:
        total.update(u)
    De = 0.0
    for k, nk in total.items():
        De += nk * (n_total - nk)
    De /= (n_total * (n_total - 1))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def alpha_dengan_selang(data, n_ulang=2000, acak=42):
    """
    Alpha beserta selang kepercayaan 95 persen melalui bootstrap unit.

    Krippendorff menekankan bahwa keputusan sebaiknya tidak diambil dari satu
    angka alpha saja, melainkan dari sebaran nilainya. Sebuah variabel baru
    layak diandalkan bila batas bawah selangnya tidak turun di bawah ambang
    minimum yang ditetapkan.

    Mengembalikan dict berisi alpha, batas bawah, batas atas, dan probabilitas
    alpha berada di bawah 0,800 dan 0,667.
    """
    rng = np.random.default_rng(acak)
    n = max(len(b) for b in data)
    titik = krippendorff_alpha(data)

    hasil = []
    for _ in range(n_ulang):
        idx = rng.integers(0, n, n)
        contoh = [[baris[i] if i < len(baris) else None for i in idx] for baris in data]
        try:
            hasil.append(krippendorff_alpha(contoh))
        except ValueError:
            continue

    hasil = np.array(hasil)
    return {
        'alpha': round(float(titik), 4),
        'batas_bawah_95': round(float(np.percentile(hasil, 2.5)), 4),
        'batas_atas_95': round(float(np.percentile(hasil, 97.5)), 4),
        'peluang_di_bawah_0800': round(float((hasil < 0.800).mean()), 4),
        'peluang_di_bawah_0667': round(float((hasil < 0.667).mean()), 4),
        'n_bootstrap': int(len(hasil)),
    }


def matriks_pelabel(label_a, label_b, kelas=None):
    """
    Matriks silang Pelabel A terhadap Pelabel B.

    Neuendorf (2017) menyarankan pemeriksaan matriks ini, bukan hanya angka
    reliabilitas tunggal, karena matriks menunjukkan pasangan kategori mana yang
    tertukar secara sistematis. Kekeliruan yang terpusat pada satu pasangan
    kategori menandakan definisi kategori yang belum tegas, bukan kecerobohan.
    """
    import pandas as pd
    a = pd.Series(label_a).astype(str).str.strip()
    b = pd.Series(label_b).astype(str).str.strip()
    valid = (a != '') & (b != '') & (a.str.lower() != 'nan') & (b.str.lower() != 'nan')
    m = pd.crosstab(a[valid], b[valid])
    if kelas:
        m = m.reindex(index=kelas, columns=kelas, fill_value=0)
    m.index.name = 'Pelabel A'
    m.columns.name = 'Pelabel B'
    return m


def tafsir(alpha):
    """Terjemahkan nilai alpha ke keputusan sesuai standar Krippendorff."""
    if alpha >= 0.800:
        return 'Dapat diandalkan. Data layak dipakai untuk menarik kesimpulan.'
    if alpha >= 0.667:
        return ('Hanya untuk kesimpulan sementara. Definisi kategori perlu '
                'dipertajam sebelum data dipakai untuk kesimpulan final.')
    return ('Belum layak. Buku kode harus diperbaiki dan pelabelan diulang '
            'pada kategori yang bermasalah.')


if __name__ == '__main__':
    # Contoh dua pelabel dengan data tidak lengkap.
    # Nilai acuan diverifikasi terhadap paket rujukan `krippendorff` (PyPI):
    # alpha nominal = 0,8522
    A = ['1', '2', '3', '3', '2', '1', '4', '1', '2', None, None, None]
    B = ['1', '2', '3', '3', '2', '2', '4', '1', '2', '5', None, '3']
    print('alpha (2 pelabel) :', round(krippendorff_alpha([A, B]), 4), '| acuan 0,8522')

    # Tiga pelabel, acuan 0,6934
    C = [None, '3', '3', '3', '2', '3', '4', '2', '2', '5', '1', '3']
    print('alpha (3 pelabel) :', round(krippendorff_alpha([A, B, C]), 4), '| acuan 0,6934')

    print('kesepakatan penuh :', krippendorff_alpha([['p', 'n', 'x'], ['p', 'n', 'x']]))
    print('berlawanan penuh  :', krippendorff_alpha([['p', 'n', 'p', 'n'],
                                                     ['n', 'p', 'n', 'p']]))
    print()
    print(tafsir(0.85))
