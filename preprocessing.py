# ============================================
# preprocessing.py - Modul Preprocessing Teks
# Smart Resto - Analisis Sentimen
# ============================================

import re
import string

# Stopwords Bahasa Indonesia (manual, tanpa NLTK)
STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'dalam', 'ada', 'juga', 'saya', 'kami',
    'kita', 'mereka', 'dia', 'ia', 'anda', 'kamu', 'sudah', 'akan',
    'hanya', 'oleh', 'atau', 'jika',
    'maka', 'karena', 'seperti', 'kalau', 'tetapi',
    'ya', 'lagi', 'pun', 'deh', 'nih', 'dong', 'lah', 'kah', 'nya', 'ku', 'mu',
    'punya', 'saat', 'waktu', 'tempat', 'cara', 'hal', 'kali', 'tahun', 'hari', 'bulan',
    # CATATAN: 'tidak','bukan','tak','jangan','kurang','sangat','banget','tapi','namun'
    # SENGAJA tidak dimasukkan stopwords karena penting untuk analisis sentimen
}

# Kamus normalisasi (slang/typo → baku)
NORMALISASI = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'gapapa': 'tidak apa', 'nggakpapa': 'tidak apa',
    'oke': 'oke', 'ok': 'oke', 'okey': 'oke',
    'mantap': 'mantap', 'mantul': 'mantap',
    'enak': 'enak', 'enakkk': 'enak', 'enak banget': 'sangat enak',
    'kece': 'bagus', 'keren': 'bagus',
    'jelek': 'buruk', 'payah': 'buruk',
    'lbh': 'lebih', 'dgn': 'dengan', 'utk': 'untuk',
    'krn': 'karena', 'tp': 'tapi', 'yg': 'yang',
    'hrs': 'harus', 'sdh': 'sudah', 'blm': 'belum',
    'smua': 'semua', 'bgt': 'banget', 'bngt': 'banget',
    'msh': 'masih', 'bs': 'bisa', 'dpt': 'dapat',
    'sy': 'saya', 'lg': 'lagi', 'jg': 'juga',
    'kl': 'kalau', 'gt': 'gitu', 'gitu': 'begitu',
    'bsk': 'besok', 'kmrn': 'kemarin', 'hari ini': 'hari ini',
    'recommended': 'rekomen', 'rekomen': 'rekomen',
    'worth': 'sebanding', 'worth it': 'sebanding',
}

# Kamus sentimen positif & negatif (lexicon)
KATA_POSITIF = {
    'enak', 'lezat', 'mantap', 'bagus', 'puas', 'suka', 'hebat',
    'rekomen', 'recommended', 'baik', 'ramah', 'cepat', 'fresh',
    'segar', 'nikmat', 'gurih', 'spesial', 'sempurna', 'memuaskan',
    'terbaik', 'istimewa', 'mewah', 'murah', 'terjangkau', 'worth',
    'crispy', 'empuk', 'lembut', 'manis', 'harum', 'wangi',
    'bersih', 'rapi', 'nyaman', 'friendly', 'helpful', 'sabar',
    'cepat', 'tepat', 'akurat', 'konsisten', 'stabil', 'oke',
}

KATA_NEGATIF = {
    'buruk', 'jelek', 'kecewa', 'lambat', 'mahal', 'tidak enak',
    'tidak suka', 'payah', 'kotor', 'basi', 'mengecewakan', 'lama',
    'dingin', 'keras', 'pahit', 'asin', 'pedas berlebihan', 'busuk',
    'bau', 'jorok', 'semrawut', 'berantakan', 'kurang', 'hambar',
    'tawar', 'tengik', 'keras', 'alot', 'mentah', 'gosong',
    'minyak', 'berminyak', 'berat', 'eneg', 'mual',
}

# Kata penguat (booster)
KATA_PENGUAT = {'sangat', 'sekali', 'banget', 'amat', 'benar', 'betul', 'bener', 'super'}
# Kata negasi
KATA_NEGASI  = {'tidak', 'bukan', 'jangan', 'tiada', 'tak', 'tanpa'}


def case_folding(text: str) -> str:
    """Ubah ke huruf kecil"""
    return text.lower().strip()


def cleaning(text: str) -> str:
    """Hapus karakter tidak relevan"""
    # Hapus URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Hapus mention & hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca (kecuali spasi)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalisasi(text: str) -> str:
    """Normalisasi kata slang/typo"""
    words = text.split()
    result = []
    for w in words:
        result.append(NORMALISASI.get(w, w))
    return ' '.join(result)


def tokenisasi(text: str) -> list:
    """Pisahkan teks menjadi token/kata"""
    return text.split()


def stopword_removal(tokens: list) -> list:
    """Hapus stopwords"""
    return [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]


# Kata-kata penting yang TIDAK boleh di-stem
KATA_PENTING = {
    # Sentimen negatif
    'kecewa', 'kecewakan', 'mengecewakan', 'kurang', 'buruk', 'jelek',
    'payah', 'basi', 'kotor', 'lambat', 'mahal', 'hambar', 'asin',
    'pahit', 'keras', 'alot', 'mentah', 'gosong', 'bau', 'jorok',
    'menyesal', 'rugi', 'bohong', 'menipu', 'tidak', 'bukan', 'tak',
    # Sentimen positif
    'enak', 'lezat', 'mantap', 'bagus', 'puas', 'suka', 'hebat',
    'ramah', 'segar', 'nikmat', 'gurih', 'sempurna', 'istimewa',
    'recommended', 'rekomen', 'worth', 'crispy', 'empuk', 'murah',
    'terjangkau', 'bersih', 'rapi', 'nyaman', 'cepat', 'fresh',
    # Kata penguat/negasi
    'sangat', 'banget', 'sekali', 'amat', 'super', 'benar',
    'terlalu', 'agak', 'lumayan', 'cukup',
}

def stemming_sederhana(word: str) -> str:
    """
    Stemming sederhana untuk Bahasa Indonesia.
    Kata penting (sentimen) tidak di-stem agar tidak rusak.
    """
    # Jangan stem kata penting
    if word in KATA_PENTING:
        return word

    # Jangan stem kata pendek
    if len(word) <= 4:
        return word

    result = word

    # Hapus imbuhan awalan (hanya jika kata cukup panjang setelah dipotong)
    prefixes = ['meng', 'meny', 'mem', 'men', 'me',
                'ber', 'ter', 'per',
                'peng', 'pem', 'pen', 'pe',
                'di', 'ke', 'se']
    for pref in prefixes:
        if result.startswith(pref) and len(result) > len(pref) + 3:
            result = result[len(pref):]
            break

    # Hapus imbuhan akhiran (hanya jika kata cukup panjang setelah dipotong)
    suffixes = ['kan', 'an', 'nya', 'lah', 'kah', 'ku', 'mu', 'i']
    for suf in suffixes:
        if result.endswith(suf) and len(result) > len(suf) + 3:
            result = result[:-len(suf)]
            break

    # Jika hasil stem terlalu pendek, kembalikan kata asli
    if len(result) < 3:
        return word

    return result


def preprocessing_pipeline(text: str) -> list:
    """
    Pipeline preprocessing lengkap:
    1. Case Folding
    2. Cleaning
    3. Normalisasi
    4. Tokenisasi
    5. Stopword Removal
    6. Stemming
    """
    text = case_folding(text)
    text = cleaning(text)
    text = normalisasi(text)
    tokens = tokenisasi(text)
    tokens = stopword_removal(tokens)
    tokens = [stemming_sederhana(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]  # filter token terlalu pendek
    return tokens


def hitung_skor_lexicon(tokens: list) -> dict:
    """
    Hitung skor sentimen berbasis lexicon
    (digunakan sebagai fallback / fitur tambahan)
    """
    skor_pos = 0
    skor_neg = 0
    prev_negasi = False
    prev_penguat = False

    for i, token in enumerate(tokens):
        is_negasi  = token in KATA_NEGASI
        is_penguat = token in KATA_PENGUAT

        if is_negasi:
            prev_negasi = True
            continue
        if is_penguat:
            prev_penguat = True
            continue

        booster = 1.5 if prev_penguat else 1.0

        if token in KATA_POSITIF:
            if prev_negasi:
                skor_neg += 1 * booster
            else:
                skor_pos += 1 * booster
        elif token in KATA_NEGATIF:
            if prev_negasi:
                skor_pos += 0.5 * booster  # negasi negatif = sedikit positif
            else:
                skor_neg += 1 * booster

        prev_negasi  = False
        prev_penguat = False

    total = skor_pos + skor_neg
    if total == 0:
        return {'positif': 0, 'negatif': 0, 'netral': 1, 'skor': 0.5}

    skor_norm = skor_pos / total  # 0 = negatif, 1 = positif

    return {
        'positif': skor_pos,
        'negatif': skor_neg,
        'netral': 1 if abs(skor_pos - skor_neg) < 0.5 else 0,
        'skor': round(skor_norm, 4)
    }


if __name__ == '__main__':
    # Test preprocessing
    contoh = [
        "Makanannya enak banget, ayam gepreknya crispy!",
        "Pelayanan lambat dan makanan tidak enak sama sekali.",
        "Biasa saja, tidak terlalu spesial tapi juga tidak mengecewakan.",
    ]
    print("=" * 60)
    print("TEST PREPROCESSING PIPELINE")
    print("=" * 60)
    for teks in contoh:
        tokens = preprocessing_pipeline(teks)
        skor   = hitung_skor_lexicon(tokens)
        print(f"\nInput  : {teks}")
        print(f"Tokens : {tokens}")
        print(f"Skor   : {skor}")