import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met
import os.path

# Ucitavanje podataka
slova = pd.read_csv(os.path.join('..', 'Podaci za rad', 'Letter-names2.csv'))

# Izdvajanje ulaznih i ciljnog atributa
ulazni = slova.columns[1:].tolist()
ciljni = 'slovo'

x = slova[ulazni]
y = slova[ciljni]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

'''
params = [{'solver':['sgd'],
           'activation' : ['logistic', 'tanh', 'relu'],
          }]

neuro_klas = GridSearchCV(MLPClassifier(), params, cv = 3)
neuro_klas.fit(x_train, y_train)

print('Najbolji parametri:')
print(neuro_klas.best_params_)

print()
'''

neuro_klas = MLPClassifier(activation = 'relu')
neuro_klas.fit(x_train, y_train)
print('Broj slojeva:', neuro_klas.n_layers_)

print('Izvestaj za test skup:')
y_pred = neuro_klas.predict(x_test)
matrica = pd.DataFrame(met.confusion_matrix(y_test, y_pred),
                       index = neuro_klas.classes_,
                       columns = neuro_klas.classes_)

print('Matrica konfuzije:', matrica, sep = '\n')

izvestaj = met.classification_report(y_test, y_pred,
                                     target_names = neuro_klas.classes_)
print('Izvestaj klasifikacije:', izvestaj, sep = '\n')
