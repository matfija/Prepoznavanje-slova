from pandas import read_csv
from os.path import join

# Ucitavanje podataka
slova = read_csv(join('..', 'Podaci za rad', 'Letter-names2.csv'))

print('Prvih pet instanci:', slova.head(), sep = '\n')

print('Opis podataka:', slova.describe(), sep = '\n')

print('Korelacija:', slova.corr(), sep = '\n')
