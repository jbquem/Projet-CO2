%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from pingouin import ancova

# Data import
df_2005 = pd.read_csv('data_co2_2005.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2006 = pd.read_csv('data_co2_2006.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2007 = pd.read_csv('data_co2_2007.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2008 = pd.read_csv('data_co2_2008.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2009 = pd.read_csv('data_co2_2009.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2010 = pd.read_csv('data_co2_2010.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2011 = pd.read_csv('data_co2_2011.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2012 = pd.read_csv('data_co2_2012.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2013 = pd.read_csv('data_co2_2013.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2014 = pd.read_csv('data_co2_2014.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')
df_2015 = pd.read_csv('data_co2_2015.csv', sep=';', encoding='latin-1', na_values = 'None',decimal=',')

# Quick data check
display(df_2005.head(2))
display(df_2006.head(2))
display(df_2007.head(2))
display(df_2008.head(2))
display(df_2009.head(2))
display(df_2010.head(2))
display(df_2011.head(2))
display(df_2012.head(2))
display(df_2013.head(2))
display(df_2014.head(2))
display(df_2015.head(2))

# %%
def drop_dup(df):
        """Permet d'afficher et de supprimer le nombre de duplicats dans les jeux de données"""
        print("Le nombre initial de duplicats est de", df['cnit'].duplicated().sum())
        df['cnit'] = df['cnit'].str.strip() # Retire les espaces en début et en fin de cellule
        df = df.drop_duplicates(subset = 'cnit', inplace = True)


drop_dup(df_2005)
drop_dup(df_2006)
drop_dup(df_2007)
drop_dup(df_2008)
drop_dup(df_2009)
drop_dup(df_2010)
drop_dup(df_2011)
drop_dup(df_2012)
drop_dup(df_2013)
drop_dup(df_2014)
drop_dup(df_2015)


# %%
data_co2 = pd.concat ([df_2005,  df_2006, df_2007, df_2008,  df_2009, df_2010, df_2011, df_2012, df_2013,\
                        df_2014, df_2015], ignore_index=True)
data_co2

# %%
# Vérification présence de duplicats après concaténation
drop_dup(data_co2)

# %%
# Vérification du type des variables
data_co2.info()

data_co2 = data_co2.astype({'puiss_max' : 'float', 'conso_urb':'float','conso_exurb':'float','conso_mixte':'float',\
                            'co_typ_1' : 'float',  'hc' : 'float', 'nox' : 'float','hcnox' : 'float',\
                            'ptcl' : 'float', 'masse_ordma_min' : 'float', 'masse_ordma_max' : 'float'})
data_co2.info()


# %%
# Vérification de la présence de NA
print(data_co2.isna().sum())

# On retire les lignes des véhicules où les taux d'émission de co2 sont manquants
data_co2 = data_co2.dropna(subset = ['co2'], axis = 0, how = 'any')

# On retire les lignes des véhicules où les informations sur la carte grise sont manquantes
data_co2 = data_co2.dropna(subset = ['cnit'], axis = 0, how = 'any')

# On retire les lignes des véhicules où les informations sur les émissions en ville sont manquantes
data_co2 = data_co2.dropna(subset = ['conso_urb'], axis = 0, how = 'any')
print(data_co2.isna().sum())




# %%
# Vérification des modalités pour les variables catégorielles

# Vérification de lib_marq_utac
display(data_co2['lib_mrq_utac'].unique())

data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].str.strip()
data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'ALFA ROMEO': 'ALFA-ROMEO', 'MERCEDES BENZ':  'MERCEDES-BENZ',\
                                                             'MERCEDES':  'MERCEDES-BENZ','FORD-CNG-TECHNIK':'FORD', \
                                                             'ROLLS ROYCE':'ROLLS-ROYCE', 'JAGUAR LAND ROVER LIMITED': 'LAND ROVER',\
                                                             'THE LONDON TAXI COMPANY': 'LTI VEHICLES', 'LADA-VAZ': 'LADA',\
                                                             'QUATTRO': 'AUDI', 'MERCEDES AMG' : 'MERCEDES-BENZ',\
                                                             'RENAULT TECH': 'RENAULT', 'ROVER': 'LAND ROVER'})

# réassignation des modalités "Dijeau Carrossier"
dijeau_carrossier = data_co2[data_co2['lib_mrq_utac'] == 'DIJEAU CARROSSIER']

for i in dijeau_carrossier['lib_mod']:
    if i == 'CADDY':
        data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'DIJEAU CARROSSIER': 'VOLKSWAGEN'})
    if i == 'TORUNEO':
        data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'DIJEAU CARROSSIER': 'FORD'})
    if i == 'KANGOO':
        data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'DIJEAU CARROSSIER': 'RENAULT'})
    if i == 'BERLINGO':
        data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'DIJEAU CARROSSIER': 'CITROEN'})
    if i == 'PARTNER':
        data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'DIJEAU CARROSSIER': 'PEUGEOT'})
    if i == 'DOBLO':
        data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'DIJEAU CARROSSIER': 'FIAT'})
                                                                
display(data_co2['lib_mrq_utac'].unique())

# Nombre de modalité pour la varibale 'lib_mrq_utac'
data_co2['lib_mrq_utac'].value_counts()

# %%
# Vérification du type de carburant
display(data_co2['cod_cbr'].unique())
data_co2['cod_cbr'] = data_co2['cod_cbr'].str.strip()
data_co2['cod_cbr'] = data_co2['cod_cbr'].replace({'ES': 'Essence', 'GO':  'Diesel', \
                                                       'ES/GN' : 'Hybride gaz naturel','GN/ES' : 'Hybride gaz naturel', 'ES/GP' : 'Essence', 'GP/ES' : 'Essence', 'GN' : 'Gaz naturel',\
                                                       'FE': 'Essence', 'EH' : 'Hybride non rechargeable', 'GH' : 'Hybride non rechargeable',\
                                                       'EE' : 'Hybride rechargeable', 'GL' : 'Hybride rechargeable', 'EN' : 'Hybride gaz naturel'})
display(data_co2['cod_cbr'].unique())



# Nombre de modalité pour la varibale carburant
data_co2['cod_cbr'].value_counts()

# %%
# Vérification de la variable hybride
display(data_co2['hybride'].unique())

data_co2['hybride'] = data_co2['hybride'].str.strip()

display(data_co2['hybride'].unique())

# Nombre de modalité pour la varibale hybride
data_co2['hybride'].value_counts()

# %%
# Vérification de la variable puissance administrative
display(data_co2['puiss_admin_98'].unique())

data_co2['puiss_admin_98'] = data_co2['puiss_admin_98'].apply(lambda x : str(x))
data_co2['puiss_admin_98'] = data_co2['puiss_admin_98'].replace('\.0', '', regex=True)

display(data_co2['puiss_admin_98'].unique())

# Nombre de modalité pour la variable puissance administrative
data_co2['puiss_admin_98'].value_counts()

# %%
# Vérification de la variable boîte de vitesses
display(data_co2['typ_boite_nb_rapp'].unique())

data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].str.strip()
data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].str.replace(" ", "")

display(data_co2['typ_boite_nb_rapp'].unique())

display(data_co2[data_co2['typ_boite_nb_rapp'] == '5'])
display(data_co2[data_co2['lib_mod'].str.contains('BRAVO 1.9 Multijet')])
# On observe que lorsqu'il y a un 5 ou un 6 pour le type de boite, il sagit d'une faute de frappe, il devrait y avoir M5 ou M6
# comme on l'observe pour les autres lignes. 
# On remplace donc le 5 par M5 et le 6 par M6

data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].replace({'6' : 'M6', '5' : 'M5'})
display(data_co2['typ_boite_nb_rapp'].unique())

display(data_co2[data_co2['typ_boite_nb_rapp'] == 'V.'])
display(data_co2[data_co2['lib_mod'].str.contains('IS 300h')])
# On observe que le type de boite est V0 pour ce modèle
display(data_co2[data_co2['lib_mod'].str.contains('FORESTER 2.0')])
# On observe que le type de boite est aussi V0 pour ce modèle
# On remplae V. par V0

data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].replace('V.', 'V0')
display(data_co2['typ_boite_nb_rapp'].unique())

# Nombre de modalité pour la variable puissance administrative
data_co2['typ_boite_nb_rapp'].value_counts()


# %%
# Vérification de la variable champ V9
display(data_co2['champ_v9'].unique())

data_co2['champ_v9'] = data_co2['champ_v9'].str.strip()
data_co2['champ_v9'] = data_co2['champ_v9'].str[-5:]
# Après vérification les véhicules '74EEV' sont des véhicules 'EURO5'
data_co2['champ_v9'] = data_co2['champ_v9'].replace({'74EEV': 'EURO5'})

display(data_co2['champ_v9'].unique())

# Nombre de modalité pour la variable champ V9
display(data_co2['champ_v9'].value_counts())

# %%
# Vérification de la variable carrosserie
display(data_co2['carrosserie'].unique())

#On enlève les mauvaises écritures de combispace et on merge les monospaces compact et monospace
data_co2['carrosserie'] = data_co2['carrosserie'].replace({"COMBISPCACE" : "COMBISPACE", "MONOSPACE COMPACT" : "MONOSPACE"})

display(data_co2['carrosserie'].unique())

# Nombre de modalité pour la variable carrosserie
display(data_co2['carrosserie'].value_counts())

# %%
# Vérification de la variable gamme
display(data_co2['gamme'].unique())

# Harmonisation des modalités
data_co2['gamme'] = data_co2['gamme'].replace({"MOY-INF": "MOY-INFER", 
                                               "MOY-INFERIEURE" : "MOY-INFER", 
                                               "ECONOMIQUE" : "INFERIEURE"})

# Avoir classe les 308 CC en coupé est une erreur
data_co2[data_co2['lib_mod'].str.contains('308 CC')]
# La bonne classe est 'moy-infer' donc on ré assigne
data_co2['gamme'] = data_co2['gamme'].replace('COUPE', 'MOY-INFER')

display(data_co2['gamme'].unique())

# %%
# Vérification de la variable étiquette énergétique

# Rajout étiquettes énergétiques
data_co2.loc[data_co2.co2 <= 100.0, 'etiq_energ'] = 'A'
data_co2.loc[data_co2.co2.between(100.1, 120.0) , 'etiq_energ'] = 'B'
data_co2.loc[data_co2.co2.between(120.1, 140.0) , 'etiq_energ'] = 'C'
data_co2.loc[data_co2.co2.between(140.1, 160.0) , 'etiq_energ'] = 'D'
data_co2.loc[data_co2.co2.between(160.1, 200.0) , 'etiq_energ'] = 'E'
data_co2.loc[data_co2.co2.between(200.1, 250.0) , 'etiq_energ'] = 'F'
data_co2.loc[data_co2.co2 > 250.0, 'etiq_energ'] = 'G'

data_co2['etiq_energ'].value_counts()

# %%
# Visualisation des données

# %%
# Emission de co2 au cours du temps

# %%
plt.figure(figsize = (100,10));
sns.catplot(x='annee_df', y='co2', kind='box', aspect = 2, data = data_co2);
plt.xlabel('\n Année', fontsize=15,  family = 'sans-serif')
plt.ylabel('Émission de CO\u2082 \n(g/km)\n', fontsize=15, family = 'sans-serif');
plt.title('Évolution des émissions de CO\u2082 du parc automobile français \n entre 2005 et 2015\n', size = 20, family = 'serif' ) # toutes les family : [ 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);

# %%
from pingouin import welch_anova, read_dataset
import scipy.stats as stats

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2, detailed = True).round(3)
display(model)

data_co2.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Emission de co2 au cours du temps en fonction du constructeur

# %%
Top_15_constructeurs = data_co2.loc[(data_co2["lib_mrq_utac"] == "MERCEDES-BENZ") | (data_co2["lib_mrq_utac"] == "VOLKSWAGEN")\
                                   | (data_co2["lib_mrq_utac"] == "AUDI") | (data_co2["lib_mrq_utac"] == "BMW")\
                                   | (data_co2["lib_mrq_utac"] == "FIAT") | (data_co2["lib_mrq_utac"] == "OPEL")\
                                   | (data_co2["lib_mrq_utac"] == "RENAULT") | (data_co2["lib_mrq_utac"] == "NISSAN")\
                                   | (data_co2["lib_mrq_utac"] == "FORD") | (data_co2["lib_mrq_utac"] == "TOYOTA")\
                                   | (data_co2["lib_mrq_utac"] == "CITROEN") | (data_co2["lib_mrq_utac"] == "PEUGEOT")\
                                   | (data_co2["lib_mrq_utac"] == "SKODA") | (data_co2["lib_mrq_utac"] == "LAMBORGHINI")\
                                   | (data_co2["lib_mrq_utac"] == "PORSCHE")]


# %%
plt.figure(figsize = (100,10));
cp = sns.catplot(x='annee_df', y='co2', kind='box', col='lib_mrq_utac', col_wrap=3,  aspect = 1.5, data = Top_15_constructeurs);
cp.fig.subplots_adjust(top=0.93) 
cp.fig.suptitle('Évolution des émissions de CO\u2082 entre 2005 et 2015', size = 35, family = 'serif')
cp.set_titles(col_template="{col_name}", size = 15)
cp.set_xlabels('\nAnnée', size = 20)
cp.set_ylabels('Émission de CO\u2082 \n(g/km)\n', size = 20)

# %%
# Vérification pour les données AUDI
data_co2_Audi = data_co2[data_co2['lib_mrq_utac'] == 'AUDI']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Audi, detailed = True).round(3)
display(model)

data_co2_Audi.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Vérification pour les données BMW
data_co2_BMW = data_co2[data_co2['lib_mrq_utac'] == 'BMW']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_BMW, detailed = True).round(3)
display(model)

data_co2_BMW.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Vérification pour les données Citroen
data_co2_Citroen = data_co2[data_co2['lib_mrq_utac'] == 'CITROEN']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Citroen, detailed = True).round(3)
display(model)

data_co2_Citroen.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Vérification pour les données Fiat
data_co2_Fiat = data_co2[data_co2['lib_mrq_utac'] == 'FIAT']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Fiat, detailed = True).round(3)
display(model)

data_co2_Fiat.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Vérification pour les données Ford
data_co2_Ford = data_co2[data_co2['lib_mrq_utac'] == 'FORD']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Ford, detailed = True).round(3)
display(model)

data_co2_Ford.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Vérification pour les données Mercedes
data_co2_Mercedes = data_co2[data_co2['lib_mrq_utac'] == 'MERCEDES-BENZ']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Mercedes, detailed = True).round(3)
display(model)

data_co2_Mercedes.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Nissan
data_co2_Nissan = data_co2[data_co2['lib_mrq_utac'] == 'NISSAN']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Nissan, detailed = True).round(3)
display(model)

data_co2_Nissan.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Opel
data_co2_Opel = data_co2[data_co2['lib_mrq_utac'] == 'OPEL']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Opel, detailed = True).round(3)
display(model)

data_co2_Opel.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Peugeot
data_co2_Peugeot = data_co2[data_co2['lib_mrq_utac'] == 'PEUGEOT']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Peugeot, detailed = True).round(3)
display(model)

data_co2_Peugeot.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Porsche
data_co2_Porsche = data_co2[data_co2['lib_mrq_utac'] == 'PORSCHE']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Porsche, detailed = True).round(3)
display(model)

data_co2_Porsche.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Renault
data_co2_Renault = data_co2[data_co2['lib_mrq_utac'] == 'RENAULT']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Renault, detailed = True).round(3)
display(model)

data_co2_Renault.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Skoda
data_co2_Skoda = data_co2[data_co2['lib_mrq_utac'] == 'SKODA']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Skoda, detailed = True).round(3)
display(model)

data_co2_Skoda.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Toyota
data_co2_Toyota = data_co2[data_co2['lib_mrq_utac'] == 'TOYOTA']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Toyota, detailed = True).round(3)
display(model)

data_co2_Toyota.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)


# %%
# Vérification pour les données Volkswagen
data_co2_Volkswagen = data_co2[data_co2['lib_mrq_utac'] == 'VOLKSWAGEN']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Volkswagen, detailed = True).round(3)
display(model)

data_co2_Volkswagen.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Vérification pour les données Lamborghini
data_co2_Lamborghini = data_co2[data_co2['lib_mrq_utac'] == 'LAMBORGHINI']

model = pg.anova(dv = 'co2', between = 'annee_df', data = data_co2_Lamborghini, detailed = True).round(3)
display(model)

data_co2_Lamborghini.pairwise_tukey(dv = 'co2', between = 'annee_df').round(3)

# %%
# Visualisation des ventes par constructeur

# %%
plt.figure(figsize = (35,10))
sns.barplot(x=data_co2['lib_mrq_utac'].value_counts().head(15).index,
            y=data_co2['lib_mrq_utac'].value_counts().head(15));
plt.xlabel('\n Nom du Constructeur', fontsize=35,  family = 'sans-serif')
plt.ylabel('Nombre de ventes\n (véhicule neuf)\n', fontsize=35, family = 'sans-serif');
plt.title('Nombre de ventes des 15 plus gros constructeurs du parc automobile français \nentre 2005 et 2015\n', size = 40, family = 'serif' ) # toutes les family : [ 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15);

# %%
# Matrice de corrélation

# %%
cor = data_co2.corr()

fig, ax = plt.subplots(figsize=(13,13))
sns.heatmap(cor, annot=True, ax=ax, cmap='magma');

# %%
# Emission de CO2 en fonction du carburant

# %%
sns.catplot(x='cod_cbr', y='co2', kind='box', data = Top_15_constructeurs);
plt.xticks(rotation = 90)
plt.ylabel('Émission de Co\u2082 \n(g/km)\n', fontsize=15,  family = 'sans-serif')
plt.xlabel('', fontsize=15, family = 'sans-serif');

# %%
# Ordinary Least Squares (OLS) model
model = pg.anova(dv = 'co2', between = 'cod_cbr', data = data_co2, detailed = True).round(3)
display(model)

data_co2.pairwise_tukey(dv = 'co2', between = 'cod_cbr').round(3)

# %%
# Nombre de véhicule vendus en fonction du carburant au cours du temps

# %%
grid = sns.FacetGrid(Top_15_constructeurs, hue = 'cod_cbr', col= 'cod_cbr', col_wrap = 3, aspect = 2, sharex=False)
grid.map(sns.countplot, 'annee_df')
grid.set_titles(col_template="{col_name}", size = 10)
plt.show()

# %%
grid = sns.FacetGrid(Top_15_constructeurs, hue = 'cod_cbr', col= 'cod_cbr', col_wrap = 3, aspect = 2, sharex=False, sharey = False)
grid.map(sns.countplot, 'annee_df')
grid.set_titles(col_template="{col_name}", size = 10)
plt.show()


