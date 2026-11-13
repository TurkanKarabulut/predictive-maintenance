# Makine Öğrenmesi Tabanlı Predictive Maintenance ve Arıza Tespit Sistemi

Makine çalışma verilerinden normal ve arızalı durumları sınıflandıran,
tahmin sonuçlarını web arayüzünde sunan bireysel mezuniyet projesi.

Bu sürüm, AI4I 2020 sentetik veri seti üzerinde geliştirilmiş bir
ara rapor prototipidir. Mevcut çalışma durumunu sınıflandırır;
gelecekteki arıza zamanını veya kalan kullanım ömrünü hesaplamaz.

## Proje bilgileri

- Öğrenci: Türkan Karabulut
- Öğrenci numarası: 23181616033
- Danışman: Doç. Dr. Saadin Oyucu
- Dönem: 2026–2027 Güz

## Tamamlanan çalışmalar

- Veri indirme, sütun ve eksik değer kontrolleri
- Veri kaynağı, lisans ve SHA-256 kaydı
- Eğitim verisinde keşifsel veri analizi ve grafikler
- Sınıf oranları korunarak eğitim/doğrulama/test ayrımı
- Ön işleme ve modeli birleştiren pipeline
- Dummy, Logistic Regression, Decision Tree ve Random Forest karşılaştırması
- Doğrulama F1 skoruna göre model seçimi
- Test değerlendirmesi ve FP/FN kayıtları
- Modelin ve sürüm bilgilerinin kaydedilmesi
- FastAPI tahmin servisi ve giriş doğrulama
- HTML/CSS/JavaScript tahmin arayüzü
- 12 başarılı otomatik test

## Kullanılan teknolojiler

Python 3.12.7, pandas, NumPy, scikit-learn, matplotlib, seaborn,
joblib, FastAPI, Pydantic, Uvicorn ve pytest.

Paket sürümleri requirements-lock.txt dosyasındadır.

## Kurulum

Windows PowerShell üzerinde proje ana klasöründe:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Python 3.12, Python başlatıcısına kayıtlı değilse ilk komutta
Python 3.12 kurulumunun python.exe yolunu kullanın.

## Uygulamayı çalıştırma

Depodaki kayıtlı modelle uygulamayı açmak için yeniden eğitim gerekmez.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --app-dir src --reload
```

- Web arayüzü: http://127.0.0.1:8000
- API dokümantasyonu: http://127.0.0.1:8000/docs
- Sağlık kontrolü: http://127.0.0.1:8000/health

Sunucuyu durdurmak için terminalde Ctrl+C kullanın.

## Veri ve deneyleri yeniden üretme

Proje ana klasöründe sırayla:

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\python.exe src\eda.py
```

Eğitim yeniden çalıştırıldığında yeni bir model sürümü oluşturulur;
metadata ve deney çıktıları güncellenir.

## Testler

```powershell
.\.venv\Scripts\python.exe -m pytest -v --junitxml=reports/api_tests.xml
```

Kaydedilmiş son test koşusunda 12 test başarılıdır.

Testler sağlık kontrolünü, kayıtlı model ile API tahmininin
eşleşmesini, geçersiz/eksik girdilerin reddini, tekrarlı tahmin
tutarlılığını ve ana sayfanın sunulmasını kapsar.

Tarayıcıdan form gönderimi ayrıca manuel olarak doğrulanmıştır.
Ana sayfa testi JavaScript akışını otomatik çalıştırmaz.

Test koşusunda HTTPX/Starlette kullanımdan kaldırma uyarısı
görülmüştür; bu uyarı henüz giderilmemiştir.

## Veri seti

AI4I 2020 Predictive Maintenance Dataset:

- Kaynak: https://archive.ics.uci.edu/dataset/601/ai4i
- DOI: https://doi.org/10.24432/C5HS5C
- Lisans: CC BY 4.0
- Veri türü: Sentetik
- Kayıt sayısı: 10.000
- Normal kayıt: 9.661
- Arızalı kayıt: 339
- Eksik değer: 0
- Tamamen tekrar eden satır: 0

Model girdileri: ürün tipi, hava sıcaklığı, işlem sıcaklığı,
dönme hızı, tork ve takım aşınmasıdır.

UDI, Product ID ve arıza türü etiketleri model girdilerine alınmamıştır.

## Deney düzeni

random_state=42 ile sınıf oranları korunarak:

- Eğitim: 6.000 kayıt
- Doğrulama: 2.000 kayıt
- Test: 2.000 kayıt

Ölçekleme ve kategorik kodlama yalnızca eğitim verisinde öğrenilir.
Öğrenen modellerde ilk dengesizlik yaklaşımı olarak
class_weight="balanced" kullanılmıştır.

Model, doğrulama F1 skoruna göre seçilmiştir. Test kümesi model
seçiminde kullanılmamıştır. Karar eşiği 0.5'tir.

## Doğrulama sonuçları

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.5054 | 0.6912 | 0.5839 | 0.9657 |
| Decision Tree | 0.2849 | 0.7794 | 0.4173 | 0.8684 |
| Logistic Regression | 0.1425 | 0.8235 | 0.2430 | 0.8788 |
| Dummy | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## Seçilen modelin test sonuçları

Seçilen model: Random Forest.

| Ölçüt | Sonuç |
|---|---:|
| Accuracy | 0.9695 |
| Precision | 0.5376 |
| Recall | 0.7353 |
| F1 | 0.6211 |
| ROC-AUC | 0.9550 |
| Average Precision | 0.6803 |
| Doğru normal | 1889 |
| Yakalanan arıza | 50 |
| Kaçırılan arıza | 18 |
| Yanlış alarm | 43 |

Sürekli normal tahmini test kümesinde 0.9660 accuracy üretir.
Bu nedenle accuracy tek başına yeterli bir başarı göstergesi değildir.

2.000 kayıt için ölçülen tek toplu tahmin süresi yaklaşık 68.82 ms'dir.
Bu değer API yanıt süresi veya tek kayıt gecikmesi değildir.

## Proje yapısı

- src/prepare_data.py: Veri edinme ve doğrulama
- src/eda.py: Eğitim verisi analizi ve grafikler
- src/train_models.py: Eğitim, seçim ve değerlendirme
- src/api.py: Tahmin API'si ve web sayfası servisi
- web/index.html: Kullanıcı arayüzü
- tests/test_api.py: Otomatik testler
- data/raw/: Ham veri ve kaynak bilgisi
- models/: Kaydedilmiş model ve metadata
- reports/: Metrikler, grafikler ve test kanıtları
- pytest.ini: Test yapılandırması

## Sınırlılıklar ve kalan işler

Sonuçlar sentetik veri ve tek bir veri bölünmesine dayalıdır.
Gerçek endüstriyel ortamda doğrulama yapılmamıştır.
Model olasılıkları kalibre edilmiş gerçek arıza riski olarak
doğrulanmamıştır.

Henüz tamamlanmayan çalışmalar:

- Çapraz doğrulama ve kontrollü hiperparametre optimizasyonu
- Boosting modeli karşılaştırması
- SHAP ile tahmin bazında açıklanabilirlik
- Veri tabanı ve kalıcı tahmin geçmişi
- Alarm yönetimi ve kapsamlı dashboard
- API gecikme, yük ve kapsamlı güvenlik testleri

## Üretken yapay zekâ kullanımı

ChatGPT; planlama, kod önerileri, hata analizi ve dokümantasyon
taslaklarında kullanılmıştır. Kodlar öğrenci tarafından yerel
ortamda çalıştırılmış; sonuçlar deney çıktıları, otomatik testler
ve manuel arayüz kontrolüyle doğrulanmıştır.