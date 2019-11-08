import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os.path

# Ucitavanje podataka
slova = pd.read_csv(os.path.join('..', 'Podaci za rad', 'Letter-names2.csv'))

# Izdvajanje ulaznih i ciljnog atributa
ulazni = slova.columns[1:].tolist()
ciljni = 'slovo'

x = slova[ulazni]
y = slova[ciljni]

# Primena PCA
pca = PCA()
x = pca.fit_transform(x)

for n in range(1, len(x[0])+1):
    # Odabir n faktora
    xk = list(map(lambda k: k[:n], x))

    # Podela skupa na trening i test u odnosu 70%-30%
    x_train, x_test, y_train, y_test = train_test_split(xk, y, test_size = 0.3)

    # Definicija raznih lenjih klasifikatora
    lenji = [(KNeighborsClassifier, {'n_neighbors': 4, 'p': 2,
                                     'weights': 'distance'}),
             (SVC, {'kernel': 'rbf', 'gamma': 'auto'})]

    # Primena svakog od njih na skup iz PCA
    for Klas, kwargs in lenji:
        klas = Klas(**kwargs)
        klas.fit(x_train, y_train)

        print('Izvestaj (%s, %d):' % (Klas.__name__, n))
        y_true, y_pred = y_test, klas.predict(x_test)
        print(classification_report(y_true, y_pred))
