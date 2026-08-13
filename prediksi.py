# ============================================
# prediksi.py - versi kompatibel model Colab
# Ganti file: C:\laragon\www\smart_resto\python\prediksi.py
# ============================================

import sys
import json
import pickle
import re
import string
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'dalam', 'ada', 'juga', 'saya', 'kami', 'kita',
    'mereka', 'dia', 'ia', 'anda', 'kamu', 'sudah', 'akan', 'hanya',
    'oleh', 'atau', 'jika', 'maka', 'karena', 'seperti', 'kalau',
    'tetapi', 'deh', 'nih', 'dong', 'lah', 'kah', 'nya', 'ku', 'mu',
    'punya', 'saat', 'waktu', 'tempat', 'cara', 'hal', 'kali',
    'tahun', 'hari', 'bulan', 'pun', 'ya', 'lagi'
}

NORMALISASI = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'gk': 'tidak', 'tdk': 'tidak', 'enakkk': 'enak', 'enakk': 'enak',
    'mantap': 'mantap', 'mantul': 'mantap', 'mantapp': 'mantap',
    'ok': 'oke', 'okay': 'oke', 'okey': 'oke',
    'bgt': 'banget', 'bngt': 'banget', 'bget': 'banget',
    'recommended': 'rekomen', 'rekomen': 'rekomen',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk',
    'krn': 'karena', 'tp': 'tapi', 'sdh': 'sudah',
    'udah': 'sudah', 'udh': 'sudah', 'bs': 'bisa',
    'sy': 'saya', 'lg': 'lagi', 'jg': 'juga',
    'kl': 'kalau', 'klo': 'kalau', 'kalo': 'kalau',
    'blm': 'belum', 'msh': 'masih', 'hrs': 'harus',
    'lbh': 'lebih', 'jd': 'jadi', 'pgn': 'pengen',
    'keren': 'bagus', 'kece': 'bagus', 'jelek': 'buruk',
    'payah': 'buruk', 'ancur': 'buruk', 'mantep': 'mantap',
    'puass': 'puas', 'kecewaaa': 'kecewa', 'nyesel': 'menyesal',
}

KATA_PENTING = {
    'enak', 'lezat', 'mantap', 'bagus', 'puas', 'suka', 'hebat',
    'rekomen', 'baik', 'ramah', 'cepat', 'fresh', 'segar', 'nikmat',
    'gurih', 'sempurna', 'memuaskan', 'terbaik', 'istimewa', 'murah',
    'terjangkau', 'crispy', 'empuk', 'lembut', 'bersih', 'nyaman',
    'kecewa', 'buruk', 'jelek', 'lambat', 'mahal', 'basi', 'kotor',
    'mengecewakan', 'dingin', 'keras', 'hambar', 'busuk', 'bau',
    'jorok', 'kurang', 'tidak', 'bukan', 'tak', 'tanpa', 'jangan',
    'sangat', 'banget', 'sekali', 'amat', 'super', 'terlalu', 'worth',
}

def _stem_sederhana(word):
    if word in KATA_PENTING or len(word) <= 4:
        return word
    prefixes = ['meng', 'meny', 'mem', 'men', 'me', 'ber', 'ter',
                'per', 'peng', 'pem', 'pen', 'pe', 'di', 'ke', 'se']
    result = word
    for pref in prefixes:
        if result.startswith(pref) and len(result) > len(pref) + 3:
            result = result[len(pref):]
            break
    suffixes = ['kan', 'an', 'nya', 'lah', 'kah', 'ku', 'mu', 'i']
    for suf in suffixes:
        if result.endswith(suf) and len(result) > len(suf) + 3:
            result = result[:-len(suf)]
            break
    return result if len(result) >= 3 else word

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = [NORMALISASI.get(w, w) for w in text.split()]
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    words = [_stem_sederhana(w) for w in words]
    return ' '.join(words)

def prediksi_lexicon(teks):
    """Fallback jika model tidak tersedia"""
    tokens = teks.lower().split()
    pos_words = {'enak','lezat','mantap','bagus','puas','suka','ramah','cepat','fresh','segar','nikmat','gurih','sempurna','terbaik','murah','terjangkau','crispy','empuk','bersih','nyaman','worth'}
    neg_words = {'kecewa','buruk','jelek','lambat','mahal','basi','kotor','mengecewakan','dingin','keras','hambar','busuk','bau','jorok','tidak enak','tidak suka'}
    pos = sum(1 for t in tokens if t in pos_words)
    neg = sum(1 for t in tokens if t in neg_words)
    if pos > neg: return {'sentimen':'positif','skor':0.7,'probabilitas':{'positif':0.7,'negatif':0.15,'netral':0.15},'metode':'lexicon'}
    elif neg > pos: return {'sentimen':'negatif','skor':0.2,'probabilitas':{'positif':0.15,'negatif':0.7,'netral':0.15},'metode':'lexicon'}
    return {'sentimen':'netral','skor':0.5,'probabilitas':{'positif':0.25,'negatif':0.25,'netral':0.5},'metode':'lexicon'}

def prediksi(teks):
    # Load model
    if not os.path.exists(MODEL_PATH):
        return prediksi_lexicon(teks)
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
    except Exception:
        return prediksi_lexicon(teks)

    vectorizer = model_data['vectorizer']
    classifier = model_data['classifier']

    teks_bersih = preprocess(teks)
    if not teks_bersih.strip():
        return prediksi_lexicon(teks)

    X = vectorizer.transform([teks_bersih])
    pred_class = classifier.predict(X)[0]
    proba = classifier.predict_proba(X)[0]
    classes = classifier.classes_

    skor_map = {cls: float(proba[i]) for i, cls in enumerate(classes)}

    if pred_class == 'positif':
        skor_final = 0.5 + skor_map.get('positif', 0.5) * 0.5
    elif pred_class == 'negatif':
        skor_final = 0.5 - skor_map.get('negatif', 0.5) * 0.5
    else:
        skor_final = 0.5

    return {
        'sentimen':     pred_class,
        'skor':         round(max(0.01, min(0.99, skor_final)), 4),
        'probabilitas': {k: round(v, 4) for k, v in skor_map.items()},
        'metode':       'random_forest',
        'tokens':       teks_bersih.split()[:10],
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Tidak ada input teks'}))
        sys.exit(1)
    teks = sys.argv[1]
    hasil = prediksi(teks)
    print(json.dumps(hasil, ensure_ascii=False))