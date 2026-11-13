# Makine Öğrenmesi Tabanlı Predictive Maintenance ve Arıza Tespit Sistemi

Makine çalışma verilerinden arıza durumunu sınıflandıran, model
kararlarını açıklayan ve sonuçları web arayüzünde sunan bireysel
mezuniyet projesi.

## Mevcut durum

- Python 3.12.7 sanal ortamı oluşturuldu.
- Veri analizi, makine öğrenmesi, API ve test paketleri kuruldu.
- Paket sürümleri requirements-lock.txt dosyasına kaydedildi.
- Paketlerin içe aktarılabildiği doğrulandı.

## Planlanan çalışmalar

- AI4I 2020 veri setinin analizi
- Veri sızıntısını önleyen model eğitim pipeline'ı
- Logistic Regression, Decision Tree ve Random Forest karşılaştırması
- Model değerlendirme ve hata analizi
- Tahmin API'si ve web arayüzü
- Açıklanabilirlik ve tahmin geçmişi

## İlk model deneyi

Veri, random_state=42 ile sınıf oranları korunarak 6000 eğitim,
2000 doğrulama ve 2000 test kaydına ayrıldı. Ön işleme yalnızca
eğitim verisinde öğrenildi.

Dummy, Logistic Regression, Decision Tree ve Random Forest
modelleri karşılaştırıldı. Doğrulama F1 skoruna göre Random
Forest seçildi. Karar eşiği 0.5 olarak kullanıldı.

Seçilen modelin test sonuçları:
- Precision: 0.5376
- Recall: 0.7353
- F1: 0.6211
- ROC-AUC: 0.9550
- Average Precision: 0.6803
- Yakalanan arıza: 50
- Kaçırılan arıza: 18
- Yanlış alarm: 43

Bunlar sentetik veri üzerinde tek bir veri bölünmesinden elde
edilen ilk sonuçlardır. Gerçek endüstriyel ortamda doğrulama
yapılmamıştır. Çapraz doğrulama ve hiperparametre optimizasyonu
henüz gerçekleştirilmemiştir.