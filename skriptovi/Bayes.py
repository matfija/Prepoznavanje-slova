import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
import os.path

# Funkcija za kumulativno mnozenje
from functools import reduce
from operator import mul
def prod(niz):
    return reduce(mul, niz, 1)

# Ucitavanje podataka
slova = pd.read_csv(os.path.join('..',
'Podaci za rad', 'Letter-names2.csv'))

# Izdvajanje ulaznih i ciljnog atributa;
# ulazni mogu bez korelisanih 2, 4, 5, 6
ulazni = slova.columns[1:].tolist()
del ulazni[0], ulazni[1:4]
ciljni = 'slovo'

# Podela na trening i test skup
train, test = train_test_split(slova, test_size = 0.3)

# Pravljenje recnika apriornih verovatnoca
klase = sorted(list(set(slova[ciljni])))

probs = {klasa: {atribut: 16*[0.00005]
                 for atribut in ulazni}
         for klasa in klase}

vrvk = {klasa: len(train[train['slovo'] == klasa]) / len(train)
        for klasa in klase}

for indeks in train.index:
    klasa = train.loc[indeks, ciljni]
    for atribut in ulazni:
        probs\
        [klasa]\
        [atribut]\
        [train.loc[indeks, atribut]] +=\
        1/(vrvk[klasa]*len(train))


# Predvidjanje klase
y_test, y_pred = [], []

for indeks in test.index:
    odabir = {klasa: vrvk[klasa] * prod([probs\
                                        [klasa]\
                                        [atribut]\
                               [test.loc[indeks, atribut]]\
                                 for atribut in ulazni])
              for klasa in klase}
    
    klasa = max(zip(odabir.values(), odabir.keys()))[1]

    y_test.append(test.loc[indeks, ciljni])
    y_pred.append(klasa)

# Ocena kvaliteta
matrica = pd.DataFrame(met.confusion_matrix(y_test, y_pred),
                       index = klase, columns = klase)
print('Matrica konfuzije:', matrica, sep = '\n')

izvestaj = met.classification_report(y_test, y_pred,
                                     target_names = klase)
print('Izvestaj klasifikacije:', izvestaj, sep = '\n')
