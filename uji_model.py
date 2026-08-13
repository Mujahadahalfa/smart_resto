# ============================================
# uji_model.py - Pengujian Model dengan Data Baru
# Smart Resto - Analisis Sentimen
# Jalankan: py uji_model.py
# ============================================

import subprocess
import json
import sys
import os

# Path Python
PYTHON = sys.executable
PREDIKSI = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediksi.py')

# ══════════════════════════════════════════════════════
# 30 KALIMAT UJI BARU (tidak ada di dataset training)
# ══════════════════════════════════════════════════════
data_uji = [
    # ── MAKANAN ──
    ("Rendangnya juara banget, daging empuk dan bumbu hitamnya kaya rempah", "positif", "makanan"),
    ("Sate padangnya lezat, saus kuningnya kental dan pedas manis", "positif", "makanan"),
    ("Es campur nya segar banget cocok untuk cuaca panas seperti ini", "positif", "makanan"),
    ("Nasinya keras seperti kerikil dan sudah tidak hangat sama sekali", "negatif", "makanan"),
    ("Sup buntutnya tidak ada rasanya, kuah bening tanpa bumbu hambar", "negatif", "makanan"),
    ("Coklatnya terlalu manis dan bikin eneg setelah beberapa suapan", "negatif", "makanan"),
    ("Menurut saya makanannya cukup oke untuk ukuran harga segini", "netral", "makanan"),
    ("Rasanya lumayan enak tapi tidak ada yang benar benar memukau", "netral", "makanan"),

    # ── PELAYANAN ──
    ("Mbaknya sangat informatif menjelaskan setiap menu dengan detail", "positif", "pelayanan"),
    ("Pesanan datang dalam waktu 5 menit padahal restoran sedang ramai", "positif", "pelayanan"),
    ("Complain saya langsung ditangani manager dengan sangat baik", "positif", "pelayanan"),
    ("Sudah panggil pelayan berkali kali tapi tidak ada yang menghampiri", "negatif", "pelayanan"),
    ("Kasirnya salah menghitung kembalian dan tidak mau mengakui kesalahan", "negatif", "pelayanan"),
    ("Staff terlihat tidak terlatih dan bingung saat ditanya soal menu", "negatif", "pelayanan"),
    ("Pelayanannya cukup standar tidak ada yang istimewa biasa saja", "netral", "pelayanan"),
    ("Kadang cepat kadang lama tergantung siapa yang bertugas melayani", "netral", "pelayanan"),

    # ── SUASANA ──
    ("Taman outdoor nya asri dengan lampu temaram sangat romantis", "positif", "suasana"),
    ("Konsep industrialnya unik dengan dinding bata dan lampu gantung", "positif", "suasana"),
    ("Toiletnya bersih dan harum selalu tersedia tisu dan sabun cuci tangan", "positif", "suasana"),
    ("Kursinya sangat tidak nyaman punggung sakit setelah duduk sejam", "negatif", "suasana"),
    ("Bau got masuk ke ruang makan membuat selera makan hilang", "negatif", "suasana"),
    ("AC bocor airnya menetes ke meja makan sangat mengganggu sekali", "negatif", "suasana"),
    ("Desain interior cukup bagus tapi kursi perlu diganti yang lebih nyaman", "netral", "suasana"),

    # ── HARGA ──
    ("Bayar seratus ribu dapat nasi ayam minuman dan dessert sangat worth", "positif", "harga"),
    ("Happy hour nya sangat menguntungkan semua minuman diskon 50 persen", "positif", "harga"),
    ("Harga lunch set nya terjangkau dan mengenyangkan cocok untuk kantong", "positif", "harga"),
    ("Segelas air mineral dihargai dua puluh ribu rupiah sangat tidak masuk akal", "negatif", "harga"),
    ("Biaya service charge tidak diberitahu di awal sangat tidak transparan", "negatif", "harga"),
    ("Harga oke tidak terlalu mahal tidak terlalu murah sesuai standar", "netral", "harga"),
    ("Untuk harga segitu masih acceptable meski bisa sedikit lebih murah", "netral", "harga"),
]

# ══════════════════════════════════════════════════════
# FUNGSI PREDIKSI
# ══════════════════════════════════════════════════════
def prediksi_kalimat(kalimat):
    try:
        result = subprocess.run(
            [PYTHON, PREDIKSI, kalimat],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        # Ambil baris JSON terakhir
        lines = [l for l in output.split('\n') if l.strip().startswith('{')]
        if lines:
            return json.loads(lines[-1])
        return None
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════
# JALANKAN PENGUJIAN
# ══════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PENGUJIAN MODEL RANDOM FOREST - DATA BARU")
print("  Smart Resto - Analisis Sentimen Ulasan Pelanggan")
print("="*65)

benar = 0
salah = 0
hasil_detail = []

emoji_kat = {
    'makanan': '🍽️', 'pelayanan': '👤',
    'suasana': '🏢', 'harga': '💰'
}
emoji_sen = {'positif': '✅', 'negatif': '❌', 'netral': '➖'}

kategori_sebelumnya = ''
for i, (kalimat, label_asli, kategori) in enumerate(data_uji, 1):
    if kategori != kategori_sebelumnya:
        print(f"\n── {emoji_kat.get(kategori,'📝')} {kategori.upper()} ──")
        kategori_sebelumnya = kategori

    hasil = prediksi_kalimat(kalimat)

    if hasil:
        pred = hasil['sentimen']
        skor = hasil['skor']
        prob = hasil['probabilitas']
        status = '✓ BENAR' if pred == label_asli else '✗ SALAH'
        if pred == label_asli:
            benar += 1
        else:
            salah += 1

        hasil_detail.append({
            'kalimat': kalimat,
            'label_asli': label_asli,
            'prediksi': pred,
            'skor': skor,
            'benar': pred == label_asli,
            'kategori': kategori,
        })

        print(f"\n  [{i:02d}] {kalimat[:58]}...")
        print(f"       Label Asli : {emoji_sen[label_asli]} {label_asli.upper()}")
        print(f"       Prediksi   : {emoji_sen[pred]} {pred.upper()}  [{status}]  skor={skor}")
        print(f"       Probabilitas: Pos={prob.get('positif',0):.3f} | Neg={prob.get('negatif',0):.3f} | Net={prob.get('netral',0):.3f}")
    else:
        print(f"\n  [{i:02d}] ERROR: tidak bisa prediksi kalimat ini")
        salah += 1

# ══════════════════════════════════════════════════════
# RINGKASAN HASIL
# ══════════════════════════════════════════════════════
total = benar + salah
akurasi = (benar / total * 100) if total > 0 else 0

print("\n" + "="*65)
print("  RINGKASAN HASIL PENGUJIAN")
print("="*65)
print(f"  Total kalimat uji : {total}")
print(f"  Prediksi benar    : {benar}")
print(f"  Prediksi salah    : {salah}")
print(f"  Akurasi pengujian : {akurasi:.2f}%")
print("="*65)

# Per kategori
print("\n  AKURASI PER KATEGORI:")
print(f"  {'Kategori':<12} {'Benar':>6} {'Total':>6} {'Akurasi':>9}")
print(f"  {'-'*38}")
for kat in ['makanan', 'pelayanan', 'suasana', 'harga']:
    kat_data = [h for h in hasil_detail if h['kategori'] == kat]
    kat_benar = sum(1 for h in kat_data if h['benar'])
    kat_total = len(kat_data)
    kat_akurasi = (kat_benar/kat_total*100) if kat_total > 0 else 0
    bar = '█' * int(kat_akurasi/10) + '░' * (10-int(kat_akurasi/10))
    print(f"  {kat:<12} {kat_benar:>6} {kat_total:>6} {kat_akurasi:>8.1f}%  {bar}")

# Per sentimen
print(f"\n  AKURASI PER SENTIMEN:")
print(f"  {'Sentimen':<12} {'Benar':>6} {'Total':>6} {'Akurasi':>9}")
print(f"  {'-'*38}")
for sen in ['positif', 'negatif', 'netral']:
    sen_data = [h for h in hasil_detail if h['label_asli'] == sen]
    sen_benar = sum(1 for h in sen_data if h['benar'])
    sen_total = len(sen_data)
    sen_akurasi = (sen_benar/sen_total*100) if sen_total > 0 else 0
    bar = '█' * int(sen_akurasi/10) + '░' * (10-int(sen_akurasi/10))
    print(f"  {sen:<12} {sen_benar:>6} {sen_total:>6} {sen_akurasi:>8.1f}%  {bar}")

# Kalimat yang salah
salah_list = [h for h in hasil_detail if not h['benar']]
if salah_list:
    print(f"\n  KALIMAT YANG SALAH DIPREDIKSI:")
    print(f"  {'-'*60}")
    for h in salah_list:
        print(f"  • {h['kalimat'][:55]}...")
        print(f"    Asli: {h['label_asli']} → Prediksi: {h['prediksi']}")

print("\n" + "="*65 + "\n")
