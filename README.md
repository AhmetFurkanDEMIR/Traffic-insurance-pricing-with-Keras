#  Traffic insurance pricing with Keras

* 128 adet taşıtın Yapay veri seti ile sigorta fiyatlandırma projesi.
* 120 eğitim verisi, 28 test verisi olarak ayırdık.
* Veri setimizin azlığı nedeniyle K-fold doğrulama ile yaklaşımımızı doğruladık.
* Veri setindeki taşıt türleri => (0 = araba), (1 = kamyon), (2 = tır) şeklinde guruplandırdık.
* Veri etiketlerini (fiyat) : araç_türü +  (yaptigi_km * 0.005) + ((100-surucu_sicil_puanı) * 2) formulü ile etiketledik.
