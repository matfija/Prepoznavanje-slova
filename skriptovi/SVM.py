import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os.path

# Ucitavanje podataka
slova = pd.read_csv(os.path.join('..', 'Podaci za rad', 'Letter-names2.csv'))

# Izdvajanje ulaznih i ciljnog atributa
ulazni = slova.columns[1:].tolist()
ciljni = 'slovo'

x = slova[ulazni]
y = slova[ciljni]

dimenzija = x.shape[1]

# Primena PCA
pca = PCA(n_components = 5)
pca.fit(x)
x_pca = pd.DataFrame(pca.transform(x),
                     columns = ['pca%d'%i for i in range(1, pca.n_components_+1)])

print('Komponente:')
for i, komponenta in zip(range(1, pca.n_components_+1), pca.components_):
    pca_opis = "pca%d ="%i
    for j, value in zip(range(0, dimenzija), komponenta):
        pca_opis += " %+.2f*%s"%(value, ulazni[j])
    print(pca_opis)

print()

# Vizuelizacija rezultata PCA
print('Objasnjena varijansa:')
for i, objr in zip(range(1, dimenzija+1), pca.explained_variance_ratio_):
    print("pca%d: %.10f"%(i, objr))

plt.figure()
plt.bar(x_pca.columns, pca.explained_variance_ratio_,
        label = 'Procenat varijanse')
plt.plot(x_pca.columns, np.cumsum(pca.explained_variance_ratio_),
         color = 'darkorange', label = 'Kumulativna varijansa',  marker = 'x')
plt.xlabel('Glavne komponente')
plt.ylabel('Udeo objasnjene varijanse')
plt.legend()
plt.show()

# Primena SVM na faktore iz PCA
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.3)
svm_klas = SVC(kernel = 'rbf', gamma = 'auto')
svm_klas.fit(x_train, y_train)

print()

print("Trening skup sa PCA:")
y_true, y_pred = y_train, svm_klas.predict(x_train)
print(classification_report(y_true, y_pred))

print("Test skup sa PCA:")
y_true, y_pred = y_test, svm_klas.predict(x_test)
print(classification_report(y_true, y_pred))

# Primena SVM na neredukovane atribute
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
svm_klas = SVC(kernel = 'rbf', gamma = 'auto')
svm_klas.fit(x_train, y_train)

print("Trening skup bez PCA:")
y_true, y_pred = y_train, svm_klas.predict(x_train)
print(classification_report(y_true, y_pred))

print("Test skup bez PCA:")
y_true, y_pred = y_test, svm_klas.predict(x_test)
print(classification_report(y_true, y_pred))

matrica = pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index = svm_klas.classes_,
                       columns = svm_klas.classes_)

print('Matrica konfuzije:', matrica, sep = '\n')
