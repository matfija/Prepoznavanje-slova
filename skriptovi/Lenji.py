import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import os.path

# Ucitavanje podataka
slova = pd.read_csv(os.path.join('..', 'Podaci za rad', 'Letter-names2.csv'))

# Izdvajanje ulaznih i ciljnog atributa;
# ulazni mogu bez korelisanih 2, 4, 5, 6
ulazni = slova.columns[1:].tolist()
del ulazni[0], ulazni[1:4]
print(ulazni)
ciljni = 'slovo'

x = slova[ulazni]
y = slova[ciljni]

# Podela skupa na trening i test u odnosu 70%-30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

'''
# Parametri za unakrsnu validacuju
params = [{'n_neighbors': range(1,10),
           'p': [1, 2],
           'weights': ['uniform', 'distance']}]

# Odabir modela
knn_klas = GridSearchCV(KNeighborsClassifier(), params)
knn_klas.fit(x_train, y_train)

print('Najbolji parametri:')
print(knn_klas.best_params_)
'''

# Definicija raznih lenjih klasifikatora
lenji = [(KNeighborsClassifier, {'n_neighbors': 4, 'p': 2,
                                'weights': 'distance'}),
         (MultinomialNB, {}),
         (GaussianNB, {})]

# Primena svakog od njih
for Klas, kwargs in lenji:
    klas = Klas(**kwargs)
    klas.fit(x_train, y_train)

    print('Izvestaj za test skup (%s):' % Klas.__name__)
    y_true, y_pred = y_test, klas.predict(x_test)
    print(classification_report(y_true, y_pred))
