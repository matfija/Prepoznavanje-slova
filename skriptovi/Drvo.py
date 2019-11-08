import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
import os.path

# Ocena modela nad skupom
def ocena(skup, pravo, predvidjeno):
    print(skup, 'skup:')
    
    print('Matrica konfuzije:')
    matrica = met.confusion_matrix(pravo, predvidjeno)
    df_matrica = pd.DataFrame(matrica,
                              index = drvo.classes_,
                              columns = drvo.classes_)
    print(df_matrica)
    
    class_report = met.classification_report(pravo, predvidjeno)
    print('Izvestaj klasifikacije:', class_report, sep = '\n')

# Vizuelizacija drveta
def vizuelizacija(drvo, ulazni, klase):
    with open('Drvo.dot', 'w') as f:
        export_graphviz(drvo, out_file = f,
                        feature_names = ulazni,
                        class_names = klase,
                        filled = True,
                        rounded = True)

# Ucitavanje podataka
slova = pd.read_csv(os.path.join('..', 'Podaci za rad', 'Letter-names2.csv'))

# Izdvajanje ulaznih i ciljnog atributa
ulazni = slova.columns[1:].tolist()
ciljni = 'slovo'

x = slova[ulazni]
y = slova[ciljni]

print('Ulazni atributi:', ulazni)
print('Ciljni atribut:', ciljni)

# Podela skupa na trening i test u odnosu 70%-30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# Instanciranje i ucenje drveta odlucivanja
drvo = DecisionTreeClassifier()
drvo.fit(x_train, y_train)

# Ispis nekih rezultata ucenja
print('Klase:', drvo.classes_)
print('Vaznost prediktora:',
      pd.Series(drvo.feature_importances_, index = ulazni),
      sep='\n')

# Vizuelizacija drveta
vizuelizacija(drvo, ulazni, y.unique())

# Primena modela na trening skup
y_pred = drvo.predict(x_train)
ocena('Trening', y_train, y_pred)

# Primena modela na test skup
y_pred = drvo.predict(x_test)
ocena('Test', y_test, y_pred)
