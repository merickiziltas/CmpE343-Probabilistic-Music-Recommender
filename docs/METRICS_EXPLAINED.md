# 📊 Part 4 Metrics - Detaylı Açıklama

## 🎯 Hit@k Metriği

### **Tanım:**
Hit@k = "İlk k ÖNERİLEN ŞARKI içinde en az bir 5★ var mı?"

### **Önemli:** k = ŞARKI SAYISI (round değil!)

### **Örnek Senaryo:**

```
Ayarlar: Her turda 5 şarkı öneriyoruz (k_per_round = 5)

┌─────────────────────────────────────────────────────────┐
│ ROUND 1 (İlk 5 şarkı):                                 │
│ ├─ Song A: 3★                                           │
│ ├─ Song B: 5★  ← İLK 5★ BURADA!                        │
│ ├─ Song C: 4★                                           │
│ ├─ Song D: 3★                                           │
│ └─ Song E: 2★                                           │
│                                                          │
│ ✅ Hit@5 = 1 (Başarılı! Song B 5★)                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ROUND 2 (6-10. şarkılar):                              │
│ ├─ Song F: 4★                                           │
│ ├─ Song G: 3★                                           │
│ ├─ Song H: 5★  ← İKİNCİ 5★                             │
│ ├─ Song I: 4★                                           │
│ └─ Song J: 3★                                           │
│                                                          │
│ ✅ Hit@10 = 1 (İlk 10 şarkıda 5★ var: Song B)          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ ROUND 3 (11-15. şarkılar):                             │
│ ├─ Song K: 3★                                           │
│ ├─ Song L: 4★                                           │
│ ├─ Song M: 3★                                           │
│ ├─ Song N: 4★                                           │
│ └─ Song O: 3★                                           │
└─────────────────────────────────────────────────────────┘

... (20 tura kadar devam)
```

### **Hit@k Hesaplama Mantığı:**

```python
# ÖRNEK: Hit@10 hesaplama

# Tüm önerilen şarkılar ve puanları
all_songs = [
    (Song A, 3), (Song B, 5), (Song C, 4), (Song D, 3), (Song E, 2),  # Round 1
    (Song F, 4), (Song G, 3), (Song H, 5), (Song I, 4), (Song J, 3),  # Round 2
    (Song K, 3), ... # Round 3+
]

# Hit@10: Sadece ilk 10 şarkıya bak
first_10 = all_songs[:10]  # İlk 10 şarkı
ratings_10 = [rating for song, rating in first_10]  # [3, 5, 4, 3, 2, 4, 3, 5, 4, 3]

# 5★ var mı?
has_5_star = (5 in ratings_10)  # True
hit_at_10 = 1 if has_5_star else 0  # 1

# ⚠️ 11. şarkıdan sonrasına BAKMAZ!
```

### **Neden Farklı k Değerleri?**

| Metrik | Anlam | Ne İçin Kullanılır? |
|--------|-------|---------------------|
| **Hit@5** | İlk 5 şarkıda 5★ var mı? | **İlk izlenim** - Kullanıcı hemen memnun oldu mu? |
| **Hit@10** | İlk 10 şarkıda 5★ var mı? | **Kısa dönem** - İki tur yeter mi? |
| **Hit@20** | İlk 20 şarkıda 5★ var mı? | **Orta dönem** - Dört tur içinde buldu mu? |

### **Gerçek Dünya Örneği:**

```
Spotify'da yeni playlist:
- Hit@5 yüksek → "Bu playlist harika!" (İlk 5 şarkı tuttu)
- Hit@5 düşük, Hit@20 yüksek → "Biraz karıştırdım ama sonra buldum"
- Hit@20 düşük → "Bu playlist bana göre değil" (Kullanıcı kaybı)
```

---

## 📈 Average Rating Metriği

### **Tanım:**
Önerilen TÜM şarkıların ortalama puanı.

### **Hesaplama:**

```python
# 20 tur × 5 şarkı = 100 şarkı önerildi

all_ratings = [
    3, 5, 4, 3, 2,  # Round 1
    4, 3, 5, 4, 3,  # Round 2
    3, 4, 3, 4, 3,  # Round 3
    ... # 20 tura kadar
]  # Toplam 100 rating

average_rating = sum(all_ratings) / len(all_ratings)
# Örnek: 378 / 100 = 3.78
```

### **Yorum:**

| Ort. Rating | Değerlendirme | Kullanıcı Durumu |
|-------------|---------------|------------------|
| **4.5 - 5.0** | Mükemmel | Çok memnun, sadık kullanıcı |
| **4.0 - 4.5** | Çok iyi | Memnun, devam edecek |
| **3.5 - 4.0** | İyi | Orta, iyileştirme gerekli |
| **3.0 - 3.5** | Orta | Risk var, bazı şarkılar kötü |
| **< 3.0** | Zayıf | Kullanıcı kaybı riski |

### **Hit@k vs. Average Rating:**

```
Model A:
  Hit@5 = 0.40 (40% kullanıcı erken 5★ buldu)
  Avg Rating = 3.60 (Genel kalite orta)
  → Stratejisi: Bazı kullanıcılar için perfect, bazıları için kötü

Model B:
  Hit@5 = 0.30 (30% kullanıcı erken 5★ buldu)
  Avg Rating = 4.10 (Genel kalite yüksek)
  → Stratejisi: Herkese iyi şarkılar, ama "wow" faktörü az
```

---

## ⏱️ Time-to-5★ Metriği (Tu)

### **Tanım:**
Kullanıcının ilk 5★'ı **hangi turda** bulduğu.

### **Part 2 ile İlişki:**
Part 2'de **Beta-Geometric** modeliyle kullanıcı sabrını (patience) modelledik.  
Time-to-5★ = Tu (Part 2'deki random variable)

### **Hesaplama:**

```python
# Her turda kontrol et
for round in range(1, 21):
    # Bu turda önerilen şarkılar
    recommended_songs = get_recommendations(round)
    
    for song, rating in recommended_songs:
        if rating == 5:
            time_to_5 = round  # İLK 5★'ın bulunduğu tur
            break
    
    if time_to_5:
        break

# Hiç bulamazsa:
if not time_to_5:
    time_to_5 = 21  # Fail case
```

### **Örnek Senaryolar:**

| Kullanıcı | Time-to-5★ | Yorum |
|-----------|------------|-------|
| Alice | 1 | **Şanslı/Sabırsız** - İlk turda buldu! |
| Bob | 3 | **Normal** - Ortalama (Part 2: E[Tu] ≈ 3.7) |
| Carol | 8 | **Seçici** - Çok aramak zorunda kaldı |
| Dave | 21 | **Bulamadı** - 20 tur boyunca hiç 5★ yok |

### **İstatistiksel Yorum:**

```
Model A: Time-to-5★ = 4.2 rounds
Model B: Time-to-5★ = 3.1 rounds

Fark: 4.2 - 3.1 = 1.1 rounds
Confidence Interval: [0.8, 1.4]

Yorum:
  → Model B, Model A'dan 1.1 tur DAHA HIZLI buldu
  → Bu istatistiksel olarak anlamlı (0 CI'da değil)
  → Model B, sabırsız kullanıcılar için daha uygun!
```

---

## 🔢 20 Turun Mantığı

### **Her Metrik İçin Gerekli Minimum:**

```
┌─────────────────────────────────────────────────┐
│ Metrik           │ Min. Şarkı │ Min. Tur       │
├─────────────────────────────────────────────────┤
│ Hit@5            │     5      │   1            │
│ Hit@10           │    10      │   2            │
│ Hit@20           │    20      │   4            │
│ Average Rating   │   100+     │  20  ← BUNUN İÇİN! │
│ Time-to-5★       │  Variable  │  20  (worst case) │
└─────────────────────────────────────────────────┘
```

### **Neden 20 Tur Optimal?**

1. **İstatistiksel Güvenilirlik:**
   - 5 tur = 25 şarkı → Az veri
   - 20 tur = 100 şarkı → Yeterli veri
   - 50 tur = 250 şarkı → Gereksiz (kim 250 şarkı dinler?)

2. **Gerçekçi Kullanıcı Davranışı:**
   ```
   Gerçek dünyada bir kullanıcı:
   - İlk 5 öneri: "Hemen denerim"
   - 10-20 öneri: "Biraz daha bakayım"
   - 50+ öneri: "Artık vaz geçtim"
   
   20 tur = Gerçekçi "vazgeçme noktası"
   ```

3. **Time-to-5★ İçin:**
   ```
   Part 2'den biliyoruz: E[Tu] ≈ 3.7 rounds
   
   Dağılım:
   - %40: 1-2 tur
   - %40: 3-5 tur
   - %15: 6-10 tur
   - %5: 11+ tur veya hiç
   
   20 tur → %95 coverage (neredeyse tüm senaryolar)
   ```

4. **Hesaplama Maliyeti:**
   ```
   100 kullanıcı × 20 tur × 5 şarkı = 10,000 öneri
   → Collaborative Filtering ile 5-10 dakika
   → Makul süre
   ```

---

## 🎯 Özet Tablo

| Soru | Cevap |
|------|-------|
| **Hit@10 yaparsam 10'dan sonraki şarkılara bakar mıyım?** | **HAYIR!** Sadece ilk 10 şarkıya bakarsın. |
| **20 tur neden gerekli?** | Average Rating (100 şarkı) ve Time-to-5★ (worst case) için. |
| **Hit@5 için 20 tur gereksiz mi?** | Evet ama diğer metrikler için lazım. Hit@5 için 1 tur yeter. |
| **k değeri ne?** | Metrikse şarkı sayısı, parametreyse tur başına öneri sayısı. |
| **Time-to-5★ nasıl hesaplanır?** | İlk 5★'ın bulunduğu TUR numarası. |

---

## 💡 Pratik Öneri

### **Konfigürasyon Örnekleri:**

```python
# Hızlı test (5 dakika)
run_simulation(n_users=50, rounds=10, k=5, hit_k_values=[5, 10])

# Standart (15 dakika)
run_simulation(n_users=100, rounds=20, k=5, hit_k_values=[5, 10, 20])

# Akademik detay (1 saat)
run_simulation(n_users=500, rounds=30, k=5, hit_k_values=[5, 10, 20, 50])
```

### **Öneri:**
Projeniz için **100 kullanıcı, 20 tur, k=5** optimal! ✅
