# Taruh file ini di: C:\laragon\www\smart_resto\python\
# Jalankan: py cek_model.py

import pickle

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

print("Tipe data model:", type(data))
print()

if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    for k, v in data.items():
        print(f"  [{k}] => {type(v).__name__}")
else:
    print("Model bukan dict, tipenya:", type(data))
    print("Attributnya:", [a for a in dir(data) if not a.startswith('_')])
