"""
Preprocessing teks bahasa Indonesia untuk analisis sentimen.

Modul ini dipakai bersama oleh notebook (pelatihan) dan aplikasi Streamlit
(inferensi), supaya teks diperlakukan dengan cara yang persis sama di kedua
tempat. Perbedaan preprocessing antara pelatihan dan inferensi adalah sumber
kesalahan yang paling sering terjadi dan paling sulit dilacak.
"""

import re

# ---------------------------------------------------------------------------
# 1. Kamus normalisasi kata tidak baku
#    Dibatasi pada kata yang benar-benar muncul di korpus DJP/pajak.
# ---------------------------------------------------------------------------
KAMUS_SLANG = {
    "yg": "yang", "yng": "yang", "dgn": "dengan", "dg": "dengan",
    "krn": "karena", "karna": "karena", "krna": "karena",
    "utk": "untuk", "untk": "untuk", "tdk": "tidak", "tp": "tapi",
    "tpi": "tapi", "tapi": "tapi", "gak": "tidak", "ga": "tidak",
    "gk": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "kagak": "tidak", "gaada": "tidak ada", "gapapa": "tidak apa apa",
    "bkn": "bukan", "blm": "belum", "blom": "belum", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "dah": "sudah", "aja": "saja",
    "aj": "saja", "sm": "sama", "sama": "sama", "org": "orang",
    "orng": "orang", "gw": "saya", "gue": "saya", "gua": "saya",
    "aku": "saya", "ane": "saya", "lu": "kamu", "lo": "kamu",
    "elu": "kamu", "km": "kamu", "kalo": "kalau", "klo": "kalau",
    "kl": "kalau", "gimana": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "kenapa": "kenapa", "bgt": "banget",
    "banget": "sangat", "bngt": "sangat", "bgtu": "begitu",
    "gitu": "begitu", "gtu": "begitu", "emang": "memang",
    "emg": "memang", "mmg": "memang", "jd": "jadi", "jgn": "jangan",
    "jangan": "jangan", "bs": "bisa", "bsa": "bisa", "dr": "dari",
    "dri": "dari", "dlm": "dalam", "pd": "pada", "sy": "saya",
    "sya": "saya", "trs": "terus", "trus": "terus", "skrg": "sekarang",
    "skrng": "sekarang", "sekarng": "sekarang", "bnyk": "banyak",
    "byk": "banyak", "hrs": "harus", "harusnya": "seharusnya",
    "sbnrnya": "sebenarnya", "sbenernya": "sebenarnya",
    "sebenernya": "sebenarnya", "bener": "benar", "bner": "benar",
    "gede": "besar", "kecil": "kecil", "duit": "uang", "duitnya": "uangnya",
    "kantor": "kantor", "pgwai": "pegawai", "pgw": "pegawai",
    "peg": "pegawai", "kmrn": "kemarin", "tgl": "tanggal",
    "thn": "tahun", "th": "tahun", "bln": "bulan", "wp": "wajib pajak",
    "ar": "account representative", "djp": "djp", "kpp": "kpp",
    "spt": "spt", "sp2dk": "sp2dk", "npwp": "npwp", "tukin": "tunjangan kinerja",
    "dll": "dan lain lain", "dsb": "dan sebagainya", "yaa": "ya",
    "sih": "", "deh": "", "dong": "", "nih": "", "tuh": "", "kok": "",
    "loh": "", "lah": "", "kan": "",
}

# ---------------------------------------------------------------------------
# 2. Kata negasi
#    TIDAK dibuang sebagai stopword, dan digabung dengan kata berikutnya.
#    "tidak bagus" -> "tidak_bagus" supaya TF-IDF tidak kehilangan negasi.
# ---------------------------------------------------------------------------
NEGASI = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang"}

# ---------------------------------------------------------------------------
# 3. Stopword
#    Daftar dasar ditulis manual supaya modul ini tetap jalan tanpa Sastrawi.
#    Kalau Sastrawi tersedia, daftarnya digabung.
#    Kata negasi sengaja dikeluarkan dari daftar stopword.
# ---------------------------------------------------------------------------
STOPWORD_DASAR = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "ini",
    "itu", "atau", "juga", "akan", "ada", "adalah", "sebagai", "oleh",
    "dalam", "saya", "kamu", "dia", "kita", "kami", "mereka", "nya",
    "saja", "sudah", "masih", "lagi", "bisa", "harus", "sangat", "lebih",
    "kalau", "jadi", "karena", "tapi", "bahwa", "agar", "supaya", "yaitu",
    "yakni", "para", "si", "se", "pun", "per", "antara", "seperti",
    "terhadap", "sampai", "hingga", "setelah", "sebelum", "ketika",
    "saat", "waktu", "orang", "banyak", "semua", "setiap", "tersebut",
    "begitu", "sini", "situ", "sana", "mana", "apa", "siapa", "kapan",
    "bagaimana", "kenapa", "mengapa", "ya", "iya", "oke", "nih", "deh",
    "dong", "kok", "loh", "sih", "aja", "gitu", "banget", "memang",
    "punya", "buat", "biar", "udah", "aku", "kan", "lah", "an",
}
STOPWORD = STOPWORD_DASAR - NEGASI

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    STOPWORD = (STOPWORD | set(StopWordRemoverFactory().get_stop_words())) - NEGASI
except Exception:
    pass

# ---------------------------------------------------------------------------
# 4. Stemmer (opsional)
# ---------------------------------------------------------------------------
_STEMMER = None


def _get_stemmer():
    global _STEMMER
    if _STEMMER is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _STEMMER = StemmerFactory().create_stemmer()
        except Exception:
            _STEMMER = False
    return _STEMMER


# ---------------------------------------------------------------------------
# 5. Istilah domain yang tidak boleh dipecah atau di-stem
# ---------------------------------------------------------------------------
ISTILAH_DOMAIN = {
    "djp", "kpp", "spt", "sp2dk", "npwp", "pph", "ppn", "pbb", "bphtb",
    "coretax", "efiling", "ebilling", "wajib", "pajak", "fiskus",
    "pemeriksaan", "tunjangan", "kinerja", "mutasi", "kemenkeu", "stan",
}

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#(\w+)")
_RE_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]+"
)
_RE_NONALFA = re.compile(r"[^a-z\s]")
_RE_SPASI = re.compile(r"\s+")
_RE_ULANG = re.compile(r"(.)\1{2,}")


def bersihkan(teks: str) -> str:
    """Tahap 1: case folding dan pembersihan karakter."""
    t = str(teks).lower()
    t = _RE_URL.sub(" ", t)
    t = _RE_MENTION.sub(" ", t)
    t = _RE_HASHTAG.sub(r" \1 ", t)
    t = _RE_EMOJI.sub(" ", t)
    t = t.replace("\u00a0", " ").replace("\u200b", " ")
    t = _RE_NONALFA.sub(" ", t)
    t = _RE_ULANG.sub(r"\1", t)          # "bangettt" -> "banget"
    t = _RE_SPASI.sub(" ", t).strip()
    return t


def normalisasi(tokens):
    """Tahap 2: ganti kata tidak baku dengan bentuk bakunya."""
    hasil = []
    for w in tokens:
        g = KAMUS_SLANG.get(w, w)
        if g:
            hasil.extend(g.split())
    return hasil


def buang_stopword(tokens):
    """Tahap 3: buang stopword, kata negasi dipertahankan."""
    return [w for w in tokens if (w not in STOPWORD and len(w) > 2) or (w in NEGASI)]


def gabung_negasi(tokens):
    """Tahap 4: satukan negasi dengan kata sesudahnya."""
    hasil, i = [], 0
    while i < len(tokens):
        if tokens[i] in NEGASI and i + 1 < len(tokens):
            hasil.append(f"{tokens[i]}_{tokens[i + 1]}")
            i += 2
        else:
            hasil.append(tokens[i])
            i += 1
    return hasil


def stem(tokens):
    """Tahap 5 (opsional): stemming, istilah domain dilewati."""
    st = _get_stemmer()
    if not st:
        return tokens
    hasil = []
    for w in tokens:
        if w in ISTILAH_DOMAIN or "_" in w:
            hasil.append(w)
        else:
            hasil.append(st.stem(w))
    return hasil


def preprocess(teks: str, pakai_stemming: bool = True) -> str:
    """Jalankan seluruh tahap dan kembalikan string siap TF-IDF."""
    tokens = bersihkan(teks).split()
    tokens = normalisasi(tokens)
    tokens = buang_stopword(tokens)
    tokens = gabung_negasi(tokens)
    if pakai_stemming:
        tokens = stem(tokens)
    return " ".join(tokens)


def preprocess_batch(daftar_teks, pakai_stemming: bool = True):
    return [preprocess(t, pakai_stemming) for t in daftar_teks]


if __name__ == "__main__":
    contoh = [
        "Gaji pegawai DJP itu gak setinggi yg kt bayangin, tp resikonya gak ringan 😔",
        "Ga lapor SPT sanksinya apa yah? Padahal udah dipotong kantor",
    ]
    for c in contoh:
        print(c)
        print("  ->", preprocess(c))
