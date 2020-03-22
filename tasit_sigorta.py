"""
veri setindeki tasit türleri => (0 = araba), (1 = kamyon), (2 = tır)

dataset in etiketlerini kod içinde hesaplıyoruz.

train_labels[i] = araba türü(0)* 300 + yaptigi_km * 0.005 + (100-surucu_sicil_puanı) * 2
şeklinde etiketler oluşturduk.

"""


import pandas as pd

veri1 = pd.read_csv('veri.csv')

veri = pd.DataFrame(veri1)

def hesapla_tasit(i):

	if i == 0:
		return 300
	
	if i == 1:
		return 400

	if i == 2:
		return 500
	

veri_tasit = [hesapla_tasit(i) for i in veri.tasit_turu]

veri_km = [i*0.005 for i in veri.yaptigi_km]

veri_kaza = [i*100 for i in veri.yaptigi_kaza]

veri_sicil = [(100-i) * 2 for i in veri.surucu_sicil_puani]

toplam = []

for i in range(len(veri.tasit_turu)):

	a = veri_tasit[i] + veri_km[i] + veri_kaza[i] + veri_sicil[i]

	toplam.append(a)

veri["fiyat"] = toplam.copy() # fiyat adlı sütun oluşturup etiketleri ekliyoruz.

train_data = veri[:100]

test_data = veri[101:129]

train_data.drop(["fiyat"],axis=1,inplace = True)

test_data.drop(["fiyat"],axis=1,inplace = True)

train_labels = veri.fiyat[:100]

test_labels = veri.fiyat[101:129]
# verileri normalize etmek
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
#her sütundan o sütunun ortalamsı çıkartılır. standart sapmasına böleriz, böylece nitelik 0 civarına ortalanaır.

test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():

  model = models.Sequential()

  model.add(layers.Dense(128, activation="relu"))
  model.add(layers.Dense(128, activation="relu"))
  model.add(layers.Dense(1)) # aktivasyon fonk. kullanmadik
  # sebebi çıktının aralığını sınırlandırmamalıyız, modelimiz ev tahmini yapacak
  
  model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

  return model


import numpy as np

k = 4

num_val_samples = len(train_data) // k

num_epochs = 600 # döngü eğitim sayisi

all_mae_histories = []


# K-fold çaprazlama

for i in range(k):

  print("işenen katman ",i)

  # k.ıncı parçadaki doğrulama verisini hazırlar.
  val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
  val_labels = train_labels[i * num_val_samples: (i+1) * num_val_samples]

  # eğitim veri setini hazırlama: veriler diğer parçalardan gelir.

  partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                      train_data[(i+1) * num_val_samples:]],
                                      axis=0)
  
  partial_train_labels = np.concatenate([train_labels[:i * num_val_samples],
                                      train_labels[(i+1) * num_val_samples:]],
                                      axis=0)

  model = build_model() # keras modelini derler

  history = model.fit(partial_train_data, partial_train_labels,epochs=num_epochs, batch_size=1, verbose=0) #model sessiz modda eğitilir, verbose = 0

  #doğrulama verisetini değerlendirir.

  val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)



  mae_history = history.history['mae']

  all_mae_histories.append(mae_history)


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# ağırlıklar rastgele oluşacak ve ilk on örnektte kötü sonuçlar çıkacaktır.
#değerlendirmeye bu örnekleri almıyoruz.
def smooth_cruve(points, factor=0.9):

  smoothed_points = []

  for point in points:

    if smoothed_points:

      previous = smoothed_points[-1]

      smoothed_points.append(previous * factor + point * (1- factor))

    else:
     
      smoothed_points.append(point)

    return smoothed_points


import matplotlib.pyplot as plt
# MAE skoru
smooth_mae_history = smooth_cruve(average_mae_history[10:])
plt.plot(range(1, len(average_mae_history) +1), average_mae_history)
plt.xlabel("Epoklar")
plt.ylabel(" MAE-(Doğrulama)")
plt.show()


test_mse_score, test_mae_score = model.evaluate(test_data,test_labels)

print(test_mae_score)

