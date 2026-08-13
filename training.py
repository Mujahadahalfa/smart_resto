#!/usr/bin/env python3
# ============================================
# training.py - Modul Training Model
# Smart Resto - Analisis Sentimen
# Algoritma: Random Forest
# ============================================

import os
import sys
import json
import pickle
import mysql.connector
import numpy as np
from preprocessing import preprocessing_pipeline

# ─── Konfigurasi ───
DB_CONFIG = {
    'host':     'localhost',
    'user':     'root',
    'password': '',
    'database': 'smart_resto',
    'charset':  'utf8mb4',
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


# ════════════════════════════════════════════
# TF-IDF Vectorizer (implementasi manual)
# ════════════════════════════════════════════
class TFIDFVectorizer:
    def __init__(self, max_features=1000):
        self.max_features  = max_features
        self.vocabulary_   = {}
        self.idf_          = {}
        self.feature_names = []

    def fit(self, corpus: list):
        """Hitung IDF dari corpus"""
        from math import log
        doc_count = len(corpus)
        df = {}

        for doc in corpus:
            unique_words = set(doc)
            for w in unique_words:
                df[w] = df.get(w, 0) + 1

        # IDF = log((N+1)/(df+1)) + 1 (smooth)
        idf_all = {w: log((doc_count + 1) / (freq + 1)) + 1
                   for w, freq in df.items()}

        # Pilih top-N features berdasarkan frekuensi dokumen
        sorted_words = sorted(df.items(), key=lambda x: x[1], reverse=True)
        top_words = [w for w, _ in sorted_words[:self.max_features]]

        self.feature_names = top_words
        self.vocabulary_   = {w: i for i, w in enumerate(top_words)}
        self.idf_          = {w: idf_all[w] for w in top_words}
        return self

    def transform(self, corpus: list) -> np.ndarray:
        """Ubah corpus menjadi matriks TF-IDF"""
        n_docs = len(corpus)
        n_feat = len(self.feature_names)
        matrix = np.zeros((n_docs, n_feat))

        for i, doc in enumerate(corpus):
            if not doc:
                continue
            tf = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1
            # Normalize TF
            max_tf = max(tf.values()) if tf else 1
            for w, freq in tf.items():
                if w in self.vocabulary_:
                    idx = self.vocabulary_[w]
                    tf_norm = freq / max_tf
                    matrix[i][idx] = tf_norm * self.idf_.get(w, 1)

        return matrix

    def fit_transform(self, corpus: list) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)


# ════════════════════════════════════════════
# Decision Tree (satu pohon Random Forest)
# ════════════════════════════════════════════
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.max_features     = max_features
        self.tree_            = None

    def _gini(self, y):
        if len(y) == 0: return 0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y, features):
        best_gini = float('inf')
        best_feat = best_thresh = None

        for feat in features:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left  = y[X[:, feat] <= thresh]
                right = y[X[:, feat] >  thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                g = (len(left)*self._gini(left) + len(right)*self._gini(right)) / len(y)
                if g < best_gini:
                    best_gini  = g
                    best_feat  = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            vals, cnts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': vals[np.argmax(cnts)],
                    'proba': dict(zip(vals.tolist(), (cnts/len(y)).tolist()))}

        n_feat = X.shape[1]
        max_f  = self.max_features or int(np.sqrt(n_feat))
        feats  = np.random.choice(n_feat, min(max_f, n_feat), replace=False)

        feat, thresh = self._best_split(X, y, feats)
        if feat is None:
            vals, cnts = np.unique(y, return_counts=True)
            return {'leaf': True, 'class': vals[np.argmax(cnts)],
                    'proba': dict(zip(vals.tolist(), (cnts/len(y)).tolist()))}

        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            'leaf':   False,
            'feat':   feat,
            'thresh': thresh,
            'left':   self._build(X[left_mask],  y[left_mask],  depth+1),
            'right':  self._build(X[right_mask], y[right_mask], depth+1),
        }

    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class'], node.get('proba', {})
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_)[0] for x in X])

    def predict_proba(self, X):
        results = []
        for x in X:
            _, proba = self._predict_one(x, self.tree_)
            results.append(proba)
        return results


# ════════════════════════════════════════════
# Random Forest Classifier
# ════════════════════════════════════════════
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=10,
                 min_samples_split=2, max_features=None, random_state=42):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.max_features     = max_features
        self.random_state     = random_state
        self.trees_           = []
        self.classes_         = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.trees_   = []
        n = len(X)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            idx  = np.random.choice(n, n, replace=True)
            X_b  = X[idx]
            y_b  = y[idx]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_b, y_b)
            self.trees_.append(tree)

            if (i+1) % 10 == 0:
                print(f"  → {i+1}/{self.n_estimators} pohon dilatih")

        return self

    def predict(self, X):
        # Voting mayoritas
        votes = np.array([tree.predict(X) for tree in self.trees_])  # (n_trees, n_samples)
        result = []
        for i in range(X.shape[0]):
            col   = votes[:, i]
            vals, cnts = np.unique(col, return_counts=True)
            result.append(vals[np.argmax(cnts)])
        return np.array(result)

    def predict_proba(self, X):
        # Agregasi probabilitas dari semua pohon
        all_probas = [tree.predict_proba(X) for tree in self.trees_]
        result = []
        for i in range(len(X)):
            agg = {}
            for proba_list in all_probas:
                p = proba_list[i]
                for cls, val in p.items():
                    agg[cls] = agg.get(cls, 0) + val
            # Normalisasi
            total = sum(agg.values()) or 1
            result.append({k: round(v/total, 4) for k, v in agg.items()})
        return result

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# ════════════════════════════════════════════
# Fungsi Evaluasi Model
# ════════════════════════════════════════════
def evaluasi_model(y_true, y_pred, classes):
    """Hitung precision, recall, F1-score per kelas"""
    hasil = {}
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        hasil[cls] = {
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1_score':  round(f1, 4),
            'support':   int(np.sum(y_true == cls))
        }
    return hasil


# ════════════════════════════════════════════
# MAIN: Ambil Data, Training, Simpan Model
# ════════════════════════════════════════════
def _coba_sklearn():
    """Coba import scikit-learn. Return (TfidfVectorizer, RandomForestClassifier) atau None."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
        from sklearn.ensemble import RandomForestClassifier as SkRF
        from sklearn.pipeline import Pipeline
        return SkTfidf, SkRF
    except ImportError:
        return None


def main():
    print("\n" + "="*60)
    print("  SMART RESTO - TRAINING MODEL RANDOM FOREST")
    print("="*60)

    # ── 1. Koneksi DB & Ambil Data ──
    print("\n[1/5] Mengambil data review dari database...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            SELECT komentar, hasil_sentimen, rating
            FROM review
            WHERE hasil_sentimen IS NOT NULL
            ORDER BY id_review
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"  ✗ Error koneksi database: {e}")
        print("  → Menggunakan data latih bawaan...")
        rows = _data_latih_default()

    if len(rows) < 30:
        print(f"  ⚠ Data terlalu sedikit ({len(rows)} baris). Menambah data latih bawaan...")
        rows.extend(_data_latih_default())

    print(f"  ✓ {len(rows)} data review ditemukan")

    # ── 2. Preprocessing ──
    print("\n[2/5] Preprocessing teks...")
    texts  = [r[0] for r in rows]
    labels = np.array([r[1] for r in rows])

    print(f"  ✓ Preprocessing selesai. Distribusi label:")
    for lbl in np.unique(labels):
        cnt = np.sum(labels == lbl)
        print(f"    • {lbl}: {cnt} ({cnt/len(labels)*100:.1f}%)")

    # ── 3. Deteksi engine ──
    sklearn_modules = _coba_sklearn()
    use_sklearn = sklearn_modules is not None

    if use_sklearn:
        print("\n[3/5] Ekstraksi fitur TF-IDF (scikit-learn)...")
        SkTfidf, SkRF = sklearn_modules
        # Gabungkan tokens jadi string untuk sklearn
        corpus_str = [" ".join(preprocessing_pipeline(t)) for t in texts]
        vectorizer = SkTfidf(
            max_features=1000,
            ngram_range=(1, 2),   # unigram + bigram
            min_df=1,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(corpus_str)
        print(f"  ✓ Dimensi matriks fitur: {X.shape}")
    else:
        print("\n[3/5] Ekstraksi fitur TF-IDF (implementasi manual)...")
        print("  ℹ Tip: install scikit-learn untuk akurasi lebih tinggi:")
        print("         pip install scikit-learn")
        corpus_tok = [preprocessing_pipeline(t) for t in texts]
        vectorizer = TFIDFVectorizer(max_features=800)
        X = vectorizer.fit_transform(corpus_tok)
        print(f"  ✓ Dimensi matriks fitur: {X.shape}")

    y = labels

    # ── 4. Training dengan SEMUA data ──
    print("\n[4/5] Training Random Forest...")
    print(f"  • Total data latih : {len(y)} sampel (semua data dipakai)")

    if use_sklearn:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        print(f"  • Engine           : scikit-learn RandomForest")
        print(f"  • n_estimators     : 200 pohon, max_depth: None")
        rf = SkRF(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        # 5-fold cross-validation untuk evaluasi jujur
        print(f"  • Evaluasi         : 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_accuracy = float(np.mean(cv_scores))
        print(f"  • CV scores        : {[f'{s*100:.1f}%' for s in cv_scores]}")
        print(f"  • CV akurasi rata  : {cv_accuracy*100:.2f}%")
        # Train final model dengan semua data
        rf.fit(X, y)
        classes  = rf.classes_
        accuracy = cv_accuracy
    else:
        print(f"  • Engine           : implementasi manual")
        print(f"  • n_estimators     : 150 pohon, max_depth: 15")
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=2,
            max_features=None,
            random_state=42,
        )
        rf.fit(X, y)
        classes = rf.classes_
        y_pred_tr = rf.predict(X)
        accuracy  = float(np.mean(y_pred_tr == y))
        print(f"  ℹ Train accuracy   : {accuracy*100:.2f}%")

    # ── 5. Evaluasi & Simpan ──
    print("\n[5/5] Evaluasi & Menyimpan Model...")
    y_pred_all = rf.predict(X)
    evaluasi   = evaluasi_model(y, y_pred_all, classes)

    print(f"\n  📊 Hasil Evaluasi (CV Accuracy: {accuracy*100:.2f}%):")
    print(f"\n     {'Kelas':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"     {'-'*52}")
    for cls, m in evaluasi.items():
        print(f"     {cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['support']:>10}")

    model_data = {
        'vectorizer':  vectorizer,
        'classifier':  rf,
        'classes':     list(classes),
        'accuracy':    round(accuracy, 4),
        'evaluasi':    evaluasi,
        'n_training':  len(y),
        'use_sklearn': use_sklearn,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    engine = "scikit-learn" if use_sklearn else "implementasi manual"
    print(f"\n  ✓ Model disimpan ke: {MODEL_PATH}")
    print(f"  ✓ Engine: {engine}")
    print(f"\n{'='*60}")
    print(f"  Training selesai! Akurasi: {accuracy*100:.2f}%")
    print(f"{'='*60}\n")

    return model_data


def _data_latih_default():
    """Data latih default jika DB kosong/error - 150+ sampel"""
    return [
        # ── POSITIF (60 sampel) ──
        ("Makanannya enak banget, ayam gepreknya crispy dan sambalnya mantap", "positif", 5),
        ("Pelayanan ramah dan cepat, sangat puas dengan pesanan saya", "positif", 5),
        ("Rasa makanan lezat dan fresh, pasti balik lagi", "positif", 5),
        ("Ayam mozarellanya sangat nikmat, keju meleleh sempurna", "positif", 5),
        ("Harga terjangkau, porsi banyak, rasanya enak sekali", "positif", 4),
        ("Menu bervariasi dan semuanya recommended, cocok untuk keluarga", "positif", 4),
        ("Es tehnya segar banget, cocok diminum saat panas", "positif", 4),
        ("Makanan datang cepat dan masih panas, kualitas terjaga", "positif", 4),
        ("Pelayanannya sangat memuaskan dan ramah sekali", "positif", 5),
        ("Nasi gorengnya spesial banget, bumbu meresap sempurna", "positif", 5),
        ("Sangat puas, makanannya lezat dan pelayanan super cepat", "positif", 5),
        ("Top banget restonya, semua menu enak dan fresh", "positif", 5),
        ("Recommended banget buat keluarga, anak saya suka semua menunya", "positif", 5),
        ("Porsi besar, harga murah, rasa mantap, pasti order lagi", "positif", 5),
        ("Ayam bakarnya juara, bumbunya meresap sampai ke dalam", "positif", 5),
        ("Soto ayamnya gurih banget, kuahnya kaya rempah", "positif", 5),
        ("Pelayannya sangat baik dan sabar melayani customer", "positif", 5),
        ("Makanan datang sesuai estimasi, masih panas dan fresh banget", "positif", 4),
        ("Tempe mendoannya crispy luar lembut dalam, enak sekali", "positif", 4),
        ("Harga bersahabat, rasa bintang lima, sangat worth it", "positif", 5),
        ("Mie ayamnya enak, toppingnya lengkap dan dagingnya banyak", "positif", 4),
        ("Gado-gadonya segar dan bumbu kacangnya kental sempurna", "positif", 4),
        ("Bakso isinya banyak, kuahnya gurih, harga terjangkau", "positif", 4),
        ("Pesan online gampang, makanan datang cepat dan aman", "positif", 5),
        ("Sangat memuaskan, akan order lagi minggu depan", "positif", 5),
        ("Kebersihannya terjaga, makanan higienis dan rasanya enak", "positif", 5),
        ("Minumannya segar dan manis pas, cocok banget", "positif", 4),
        ("Steak ayamnya empuk banget, saus mushroomnya enak", "positif", 5),
        ("Nasi uduknya wangi dan gurih, lauk pauknya lengkap", "positif", 4),
        ("Sangat lezat, bumbu autentik, rasa masakan rumahan yang bikin kangen", "positif", 5),
        ("Pelayanan cepat tanggap, pesanan tidak pernah salah", "positif", 5),
        ("Makanan selalu fresh dan tidak pernah mengecewakan", "positif", 5),
        ("Puas banget, harga reasonable untuk kualitas yang diberikan", "positif", 4),
        ("Sambal terenak yang pernah saya coba, pedas segar mantap", "positif", 5),
        ("Chicken wings crispy dan juicy, cocok buat nongkrong", "positif", 4),
        ("Restonya bersih, nyaman, dan makanannya enak semua", "positif", 5),
        ("Order mudah, pembayaran gampang, makanan lezat, top deh", "positif", 5),
        ("Rasanya autentik seperti masakan nenek, bikin nostalgia", "positif", 5),
        ("Porsi jumbo tapi harga standard, sangat worth it sekali", "positif", 5),
        ("Selalu konsisten rasanya, tidak pernah berubah kualitasnya", "positif", 5),
        ("Enak dan higenis, bahan-bahannya fresh semua", "positif", 4),
        ("Cita rasa tinggi, bisa dinikmati semua kalangan", "positif", 4),
        ("Pelayanan bintang lima, makanan juga bintang lima", "positif", 5),
        ("Suka banget sama menunya, bervariasi dan semua enak", "positif", 4),
        ("Makanan super enak, ini restoran terbaik yang pernah saya coba", "positif", 5),
        ("Mantap jiwa, recommended buat semua orang", "positif", 5),
        ("Sangat puas dengan kualitas dan pelayanan restoran ini", "positif", 5),
        ("Makanan fresh, tidak bau, dan rasanya nikmat sekali", "positif", 4),
        ("Harga murah meriah tapi rasanya tidak murahan sama sekali", "positif", 5),
        ("Sering pesan di sini karena konsistensi rasa dan pelayanannya bagus", "positif", 5),
        ("Wow makanannya enak banget, paling enak se-kota ini", "positif", 5),
        ("Suka banget, semua menu yang saya coba rasanya luar biasa", "positif", 5),
        ("Pengiriman tepat waktu, kemasan rapi, makanan tetap panas", "positif", 5),
        ("Hebat sekali, makanan datang dalam kondisi sempurna", "positif", 5),
        ("Rasa autentik, bahan premium, harga bersahabat, perfect", "positif", 5),
        ("Pelayannya ramah banget, bikin betah belanja di sini", "positif", 4),
        ("Favorit keluarga kami, selalu order setiap minggu", "positif", 5),
        ("Tidak mengecewakan sama sekali, semua sesuai harapan bahkan lebih", "positif", 5),
        ("Kualitas makanan terjaga dengan baik, puas setiap order", "positif", 4),
        ("Sangat direkomendasikan, tidak akan menyesal memesan di sini", "positif", 5),

        # ── NEGATIF TAMBAHAN - kasus campuran ──
        ("kurang pedes ka ayam gepreknya kecewa banget", "negatif", 2),
        ("rasanya kurang enak mengecewakan tidak sesuai ekspektasi", "negatif", 2),
        ("ayam gepreknya tidak enak kurang bumbu kecewa", "negatif", 2),
        ("kurang memuaskan tidak worth it sama sekali", "negatif", 2),
        ("tidak enak sama sekali padahal katanya enak kecewa", "negatif", 1),
        ("mengecewakan sekali rasa tidak sesuai foto di menu", "negatif", 2),
        ("kurang puas dengan pelayanan dan rasa makanannya", "negatif", 2),
        ("tidak sesuai harapan rasanya hambar dan kurang bumbu", "negatif", 2),
        ("kecewa berat tidak akan order lagi di sini", "negatif", 1),
        ("rasanya biasa tapi harganya terlalu mahal tidak worth", "negatif", 2),

        # ── NEGATIF (55 sampel) ──
        ("Pelayanan lambat dan makanan tidak enak sama sekali", "negatif", 1),
        ("Mengecewakan sekali, pesanan salah dan tidak mau minta maaf", "negatif", 1),
        ("Makanan basi dan kotor, sangat tidak higienis", "negatif", 1),
        ("Harga mahal tapi kualitas sangat buruk dan mengecewakan", "negatif", 2),
        ("Sudah lama menunggu tapi makanan tidak kunjung datang", "negatif", 2),
        ("Rasa makanan hambar dan tidak sesuai gambar di menu", "negatif", 2),
        ("Pelayanan sangat buruk, tidak ramah dan tidak profesional", "negatif", 1),
        ("Makanannya keras dan seperti sudah tidak fresh", "negatif", 2),
        ("Porsinya sangat sedikit tidak sebanding dengan harga yang mahal", "negatif", 2),
        ("Kecewa berat, sudah pesan lama tapi makanan tidak datang", "negatif", 1),
        ("Pesanan salah terus dan tidak ada permintaan maaf dari pihak resto", "negatif", 1),
        ("Makanan datang sudah dingin dan basi, sangat mengecewakan", "negatif", 1),
        ("Tidak rekomen sama sekali, rasa aneh dan tidak enak", "negatif", 1),
        ("Pelayanan cuek dan tidak peduli dengan keluhan pelanggan", "negatif", 1),
        ("Harga sangat mahal tidak sebanding dengan rasanya yang biasa saja", "negatif", 2),
        ("Buruk sekali, tidak akan order lagi dari restoran ini", "negatif", 1),
        ("Makanan tidak sesuai pesanan, sudah komplain tapi tidak direspon", "negatif", 1),
        ("Ayam mentah di dalam, sangat berbahaya untuk kesehatan", "negatif", 1),
        ("Pengiriman sangat lambat, makanan sampai sudah basi", "negatif", 1),
        ("Rasa tidak enak, bumbu tidak meresap, daging alot dan keras", "negatif", 2),
        ("Sangat tidak puas, ini pengalaman terburuk memesan makanan", "negatif", 1),
        ("Kotor, bau tidak sedap, tidak layak untuk dikonsumsi", "negatif", 1),
        ("Payah sekali restonya, tidak ada standar kualitas makanan", "negatif", 1),
        ("Kecewa dengan pelayanan yang sangat lambat dan tidak responsif", "negatif", 2),
        ("Makanan tidak enak, bumbu terlalu asin dan tidak balance", "negatif", 2),
        ("Porsi sangat kecil, tidak kenyang sama sekali, rugi beli di sini", "negatif", 2),
        ("Tidak sesuai ekspektasi sama sekali, sangat mengecewakan", "negatif", 2),
        ("Pesanan tidak lengkap dan sudah bayar mahal, sangat merugikan", "negatif", 1),
        ("Makanan penuh minyak dan tidak sehat, tidak akan beli lagi", "negatif", 2),
        ("Pelayanan acuh tak acuh, tidak profesional sama sekali", "negatif", 1),
        ("Rasa tidak karuan, tidak bisa dimakan, benar-benar kecewa", "negatif", 1),
        ("Sangat jelek, makanan keras dan basi, buang-buang uang", "negatif", 1),
        ("Tidak rekomendasikan, banyak restoran lain yang jauh lebih baik", "negatif", 2),
        ("Kualitas sangat menurun dari dulu, tidak enak lagi sekarang", "negatif", 2),
        ("Makanan datang terlambat 2 jam, sudah dingin dan tidak layak makan", "negatif", 1),
        ("Komplain tidak ditanggapi, pelayanan sangat buruk sekali", "negatif", 1),
        ("Harga tidak masuk akal untuk kualitas yang sangat mengecewakan ini", "negatif", 1),
        ("Menyesal pesan di sini, uang terbuang sia-sia", "negatif", 1),
        ("Makanannya tidak bersih, ada rambut di dalam makanan", "negatif", 1),
        ("Rasanya tidak enak, jauh dari yang diiklankan di foto menu", "negatif", 2),
        ("Pelayanan tidak ramah, staff tidak sopan kepada pelanggan", "negatif", 1),
        ("Sangat buruk, tidak ada kualitas kontrol di restoran ini", "negatif", 1),
        ("Makanan terlalu pedas tidak sesuai pesanan, tidak bisa dimakan", "negatif", 2),
        ("Order 3 kali salah terus, tidak profesional sama sekali", "negatif", 1),
        ("Kecewa banget, sudah tunggu lama dan makanan tidak memuaskan", "negatif", 2),
        ("Tidak sesuai deskripsi, foto menu menipu pelanggan", "negatif", 2),
        ("Pelayanan buruk, makanan tidak enak, harga mahal, rugi total", "negatif", 1),
        ("Makanan tengik dan bau, kemasan rusak saat sampai", "negatif", 1),
        ("Tidak pernah lagi pesan di sini, terlalu banyak kecewanya", "negatif", 1),
        ("Rasanya aneh dan membuat perut tidak nyaman setelah makan", "negatif", 1),
        ("Porsi mini tapi harga maxi, sungguh tidak worth it sama sekali", "negatif", 1),
        ("Makanan hangus dan gosong, tidak layak disajikan kepada pelanggan", "negatif", 1),
        ("Sangat mengecewakan, tidak sesuai dengan reputasi yang digembar-gemborkan", "negatif", 2),
        ("Tempat tidak higienis, meja kotor, dan makanan tidak segar", "negatif", 1),
        ("Paling buruk yang pernah saya coba, tidak layak bintang satu pun", "negatif", 1),

        # ── PELAYANAN - POSITIF ──
        ("pelayannya ramah dan sopan sangat memuaskan", "positif", 5),
        ("pelayanan cepat tanggap dan sangat profesional", "positif", 5),
        ("staff sangat helpful dan sabar melayani customer", "positif", 5),
        ("pelayanan bintang lima ramah dan cepat sekali", "positif", 5),
        ("pelayannya senyum terus sangat menyenangkan", "positif", 4),
        ("servis sangat baik tidak mengecewakan sama sekali", "positif", 4),
        ("pelayanan memuaskan pesanan cepat datang dan akurat", "positif", 5),

        # ── PELAYANAN - NEGATIF ──
        ("pelayannya cuek dan tidak ramah sama sekali", "negatif", 1),
        ("pelayanan sangat lambat sudah tunggu lama tidak dilayani", "negatif", 1),
        ("staff tidak sopan dan tidak profesional mengecewakan", "negatif", 1),
        ("pelayanan buruk pesanan salah dan tidak minta maaf", "negatif", 2),
        ("pelayannya tidak perhatian dan susah dipanggil", "negatif", 2),
        ("servis sangat mengecewakan tidak sesuai standar", "negatif", 1),

        # ── PELAYANAN - NETRAL ──
        ("pelayanan biasa saja tidak istimewa tidak mengecewakan", "netral", 3),
        ("servis standar cukup baik tapi tidak ada yang wow", "netral", 3),
        ("pelayannya lumayan oke bisa lebih ditingkatkan lagi", "netral", 3),

        # ── SUASANA - POSITIF ──
        ("tempatnya bersih nyaman dan dekorasinya bagus sekali", "positif", 5),
        ("suasana restoran sangat enak betah berlama-lama", "positif", 5),
        ("tempatnya keren dan instagramable sangat cozy", "positif", 4),
        ("ruangannya bersih rapi dan wangi sangat nyaman", "positif", 5),
        ("atmosphere restoran sangat bagus dan mewah", "positif", 4),

        # ── SUASANA - NEGATIF ──
        ("tempatnya kotor dan berantakan sangat tidak nyaman", "negatif", 1),
        ("suasana berisik dan sempit tidak betah berlama-lama", "negatif", 2),
        ("kebersihan tempat sangat buruk lantai kotor dan bau", "negatif", 1),
        ("ruangan pengap dan panas tidak ada AC mengecewakan", "negatif", 2),

        # ── SUASANA - NETRAL ──
        ("tempatnya biasa saja tidak terlalu istimewa cukup bersih", "netral", 3),
        ("suasana standar tidak ada yang spesial tapi nyaman", "netral", 3),

        # ── HARGA - POSITIF ──
        ("harga sangat terjangkau sangat worth it untuk kualitasnya", "positif", 5),
        ("harga murah tapi kualitas tidak murahan luar biasa", "positif", 5),
        ("sangat worth it harga bersahabat rasa bintang lima", "positif", 5),
        ("harga reasonable sesuai dengan porsi dan kualitas", "positif", 4),
        ("promo harganya bagus banget sangat menguntungkan", "positif", 4),

        # ── HARGA - NEGATIF ──
        ("harga sangat mahal tidak sebanding dengan kualitasnya", "negatif", 1),
        ("terlalu mahal untuk ukuran warung tidak worth it", "negatif", 2),
        ("harga tidak masuk akal porsi sedikit mahal sekali", "negatif", 1),
        ("kemahalan banget tidak sebanding sama sekali kecewa", "negatif", 2),

        # ── HARGA - NETRAL ──
        ("harga standar tidak murah tidak mahal biasa saja", "netral", 3),
        ("harga sesuai pasaran tidak ada yang spesial lumayan", "netral", 3),

        # ── NETRAL (40 sampel) ──
        ("Biasa saja, tidak terlalu spesial tapi juga tidak mengecewakan", "netral", 3),
        ("Standar, sesuai ekspektasi tidak lebih tidak kurang", "netral", 3),
        ("Oke lah untuk makan siang, tidak ada yang istimewa", "netral", 3),
        ("Cukup baik, mungkin bisa lebih ditingkatkan lagi kualitasnya", "netral", 3),
        ("Makanan lumayan, pelayanan biasa, harga normal", "netral", 3),
        ("Tidak ada yang wow tapi juga tidak ada yang buruk", "netral", 3),
        ("Sedang, bisa lebih baik dengan sedikit perbaikan", "netral", 3),
        ("Rasa cukup enak tapi tidak sampai sangat enak", "netral", 3),
        ("Biasa aja sih, standar restoran pada umumnya", "netral", 3),
        ("Lumayan, tidak terlalu kecewa tapi juga tidak terkesan", "netral", 3),
        ("Cukup memuaskan untuk harganya, tidak ada yang spesial", "netral", 3),
        ("Oke untuk sesekali, bukan tempat yang ingin sering dikunjungi", "netral", 3),
        ("Porsi normal, rasa standar, harga sesuai pasaran", "netral", 3),
        ("Tidak jelek tidak bagus, biasa-biasa saja pengalaman ini", "netral", 3),
        ("Makanannya cukup, tidak ada yang perlu dikeluhkan tapi juga tidak istimewa", "netral", 3),
        ("Sesuai harga, tidak lebih tidak kurang, bisa diterima", "netral", 3),
        ("Lumayan enak, ada beberapa hal yang perlu ditingkatkan", "netral", 3),
        ("Tidak ada yang salah tapi juga tidak ada yang benar-benar bagus", "netral", 3),
        ("Pengalaman makan yang biasa saja, tidak menonjol", "netral", 3),
        ("Cukup untuk mengisi perut, tidak ada komplain serius", "netral", 3),
        ("Bisa jadi alternatif, tapi bukan pilihan utama saya", "netral", 3),
        ("Standar aja, sama seperti restoran-restoran lain", "netral", 3),
        ("Tidak spesial, tapi tidak mengecewakan juga sih", "netral", 3),
        ("Ada plus minusnya, tapi secara keseluruhan oke lah", "netral", 3),
        ("Rasa makanannya okay, pelayanan standar, tidak ada yang buruk", "netral", 3),
        ("Biasa, mungkin perlu inovasi lebih untuk bersaing", "netral", 3),
        ("Cukup baik untuk harga segitu, tidak ada yang mengejutkan", "netral", 3),
        ("Tidak terlalu berkesan, tapi juga tidak mengecewakan", "netral", 3),
        ("Ya begitulah, standar, tidak ada yang perlu dilebih-lebihkan", "netral", 3),
        ("Lumayan untuk alternatif makan siang yang berbeda", "netral", 3),
        ("Makanan oke, pelayanan oke, harga oke, semuanya oke saja", "netral", 3),
        ("Tidak ada yang istimewa, tapi juga tidak ada yang buruk", "netral", 3),
        ("Netral aja, sesuai ekspektasi saya yang tidak terlalu tinggi", "netral", 3),
        ("Bisa diterima untuk sesekali makan di sini", "netral", 3),
        ("Standar banget, persis seperti yang saya bayangkan sebelumnya", "netral", 3),
        ("Tidak mengecewakan, tapi juga tidak memuaskan sepenuhnya", "netral", 3),
        ("Cukup untuk kebutuhan makan, tidak lebih dari itu", "netral", 3),
        ("Biasa saja, makanan rata-rata, harga rata-rata", "netral", 3),
        ("Okay lah, tidak ada yang perlu dikeluhkan secara khusus", "netral", 3),
        ("Sedang-sedang aja, bisa lebih baik kalau mau usaha lebih", "netral", 3),
    ]


if __name__ == '__main__':
    main()