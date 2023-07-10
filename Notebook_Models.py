# %%
# Préparation du jeu de données
import pandas as pd
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

data_co2 = pd.concat ([df_2005,  df_2006, df_2007, df_2008,  df_2009, df_2010, df_2011, df_2012, df_2013,\
                        df_2014, df_2015], ignore_index=True)

def drop_dup(df):
        """Permet d'afficher et de supprimer le nombre de duplicats dans les jeux de données"""
        df['cnit'] = df['cnit'].str.strip() # Retire les espaces en début et en fin de cellule
        df = df.drop_duplicates(subset = 'cnit', inplace = True)
        
drop_dup(data_co2)

data_co2['cnit'].nunique()
data_co2['cnit'] = data_co2['cnit'].str.strip()
data_co2['cnit'].nunique()

data_co2 = data_co2.astype({'conso_urb':'float','conso_exurb':'float',\
                            'conso_mixte':'float', 'hc' : 'float', 'nox' : 'float','hcnox' : 'float',\
                            'ptcl' : 'float', 'co_typ_1' : 'float', 'puiss_max' : 'float'})

data_co2 = data_co2.dropna(subset = ['cnit'], how = 'any')
data_co2 = data_co2.dropna(subset = ['co2'], how = 'any')
data_co2 = data_co2.dropna(subset = ['conso_urb'], how = 'any')

data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].str.strip()

data_co2['lib_mrq_utac'] = data_co2['lib_mrq_utac'].replace({'ALFA ROMEO': 'ALFA-ROMEO', 'MERCEDES BENZ':  'MERCEDES-BENZ',
                                                             'FORD-CNG-TECHNIK':'FORD', 'ROLLS ROYCE':'ROLLS-ROYCE',
                                                             'THE LONDON TAXI COMPANY': 'LTI VEHICLES', 'LADA-VAZ': 'LADA',
                                                             'QUATTRO': 'AUDI', 'MERCEDES AMG' : 'MERCEDES-BENZ',
                                                             'RENAULT TECH': 'RENAULT', 'MERCEDES' : 'MERCEDES-BENZ',
                                                             'JAGUAR LAND ROVER LIMITED' : 'LAND ROVER', 'ROVER' : 'LAND ROVER',})

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
        
data_co2['cod_cbr'] = data_co2['cod_cbr'].str.strip()
data_co2['cod_cbr'] = data_co2['cod_cbr'].replace({'ES': 'Essence', 'GO':  'Diesel', \
                                                       'ES/GN' : 'Hybride gaz naturel','GN/ES' : 'Hybride gaz naturel', 'ES/GP' : 'Essence', 'GP/ES' : 'Essence', 'GN' : 'Gaz naturel',\
                                                       'FE': 'Essence', 'EH' : 'Hybride non rechargeable', 'GH' : 'Hybride non rechargeable',\
                                                       'EE' : 'Hybride rechargeable', 'GL' : 'Hybride rechargeable', 'EN' : 'Hybride gaz naturel'})
data_co2['hybride'] = data_co2['hybride'].str.strip()

data_co2['puiss_admin_98'] = data_co2['puiss_admin_98'].apply(lambda x : str(x))
data_co2['puiss_admin_98'] = data_co2['puiss_admin_98'].replace('\.0', '', regex=True)

data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].str.strip()
data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].str.replace(" ", "")

data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].replace({'6' : 'M6', '5' : 'M5'})
data_co2['typ_boite_nb_rapp'] = data_co2['typ_boite_nb_rapp'].replace('V.', 'V0')

data_co2['champ_v9'] = data_co2['champ_v9'].str.strip()
data_co2['champ_v9'] = data_co2['champ_v9'].str[-5:]

data_co2['carrosserie'] = data_co2['carrosserie'].replace({"COMBISPCACE" : "COMBISPACE", "MONOSPACE COMPACT" : "MONOSPACE"})

data_co2['gamme'] = data_co2['gamme'].replace({"MOY-INF": "MOY-INFER", 
                                               "MOY-INFERIEURE" : "MOY-INFER", 
                                               "ECONOMIQUE" : "INFERIEURE"})

data_co2[data_co2['lib_mod'].str.contains('308 CC')]  
data_co2['gamme'] = data_co2['gamme'].replace('COUPE', 'MOY-INFER')

data_co2.loc[data_co2.co2 <= 100.0, 'etiq_energ'] = 'A'
data_co2.loc[data_co2.co2.between(100.1, 120.0) , 'etiq_energ'] = 'B'
data_co2.loc[data_co2.co2.between(120.1, 140.0) , 'etiq_energ'] = 'C'
data_co2.loc[data_co2.co2.between(140.1, 160.0) , 'etiq_energ'] = 'D'
data_co2.loc[data_co2.co2.between(160.1, 200.0) , 'etiq_energ'] = 'E'
data_co2.loc[data_co2.co2.between(200.1, 250.0) , 'etiq_energ'] = 'F'
data_co2.loc[data_co2.co2 > 250.0, 'etiq_energ'] = 'G'


# %%
#Nettoyage des données

# On retire les varibales lib_mod, cnit et lib_mrq_utac car encodage compliqué
data_co2_model = data_co2.drop('lib_mod', axis = 1)
data_co2_model = data_co2_model.drop('cnit', axis = 1)
data_co2_model = data_co2_model.drop('lib_mrq_utac', axis = 1) 

# On retire la variable hc car beaucoup trop de NA
# On retire la variable bonus-malus car beaucoup trop de NA
# On retire la variable nox car redondante avec hcnox
data_co2_model = data_co2_model.drop(['hc', 'nox', 'bonus_malus'], axis = 1)

# On retire les NA de hcnox et de carrosserie et particule
data_co2_model = data_co2_model.dropna(subset = ['hcnox', 'carrosserie', 'ptcl'])

# Réduction des modalités similaires pour cod_cbr, type_noite_nb_rapp, champ_v9 et carrosserie
data_co2_model['cod_cbr'] = data_co2_model['cod_cbr'].replace ('GL', 'GH')
data_co2_model['typ_boite_nb_rapp'] = data_co2_model['typ_boite_nb_rapp'].replace({'M6' : 'M', 'M5' : 'M',
                                                                                   'A6' : 'A', 'A4' : 'A', 'A7' : 'A',
                                                                                   'V0' : 'A', 'D6' : 'A', 'A8' : 'A',
                                                                                   'D5' : 'A', 'S6' : 'A', 'D5' : 'A',
                                                                                   'A9' : 'A', 'A5' : 'A'})

names_to_keep =  ['EURO5', 'EURO6']
data_co2_model = data_co2_model[data_co2_model['champ_v9'].isin(names_to_keep)]

data_co2_model['carrosserie'] = data_co2_model['carrosserie'].replace({'COUPE' : 'COUPE_CARBIOLET', \
                                                                       'CABRIOLET' : 'COUPE_CARBIOLET', \
                                                                       'COMBISPACE' : 'MONOSPACE'})


# %%
# Encodage ordinal de la variable etiq_energ
data_co2_model['etiq_energ'] = data_co2_model['etiq_energ'].replace({'G': 0, 'F': 1, 'E': 2, \
                                                                     'D': 3, 'C': 4, 'B': 5, 'A': 6})


# %%
# Harmonisation du type pour les variable puiss_admin_98 et annee_df
data_co2_model['puiss_admin_98'] = data_co2_model['puiss_admin_98'].astype(float)
data_co2_model['annee_df'] = data_co2_model['annee_df'].astype(object)
data_co2_model.info()

# %%
# Création d'une variable quantile CO2 qui sera utilisée pour vérifier les différences observées entre 
# les modèles CO2 et les modèles étiquettes énergétiques
data_co2_model['quantile_co2'] = None

print("Quantiles :", data_co2_model['co2'].quantile([.25, .5, .75]))
# Quantiles = 193 ; 203 ; 216

# CLassifiction sur variable CO2 par rapport au quantile
data_co2_model.loc[data_co2_model['co2'] < 193.0, 'quantile_co2'] = 3
data_co2_model.loc[data_co2_model['co2'].between(193, 203) , 'quantile_co2'] = 2
data_co2_model.loc[data_co2_model['co2'].between(203.1, 216) , 'quantile_co2'] = 1
data_co2_model.loc[data_co2_model['co2'] > 216.0, 'quantile_co2'] = 0

data_co2_model['quantile_co2'] = data_co2_model['quantile_co2'].astype('int')
data_co2_model.head()

# %%
# Encodage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %%
# Création des variables cibles

#Les variables conso sont également supprimées pour éviter des problèmes de redondances avec la variable co2
to_drop = ['etiq_energ', 'conso_urb', 'conso_exurb', 'conso_mixte', 'co2', 'quantile_co2']
feats = data_co2_model.drop(to_drop, axis=1)

# Variable cible : étiquette énergétique
target_energ = data_co2_model['etiq_energ']

# Variable cible : co2
target_co2 = data_co2_model['co2']
target_co2_quantile = data_co2_model['quantile_co2']

# %%
# Création des jeux d'entraînement et des jeux test

X_train_energ, X_test_energ, y_train_energ, y_test_energ = train_test_split(feats, target_energ, \
                                                                            test_size = 0.26, random_state = 42)

X_train_co2, X_test_co2, y_train_co2, y_test_co2 = train_test_split(feats, target_co2, \
                                                                    test_size = 0.26, random_state = 42)

X_train_co2_q, X_test_co2_q, y_train_co2_q, y_test_co2_q = train_test_split(feats, target_co2_quantile, \
                                                                    test_size = 0.26, random_state = 42)

print("Train Set energ:", X_train_energ.shape)
print("Test Set energ:", X_test_energ.shape)

print("Train Set CO2:", X_train_co2.shape)
print("Test Set CO2:", X_test_co2.shape)

print("Train Set CO2 quantile:", X_train_co2_q.shape)
print("Test Set CO2 quantile:", X_test_co2_q.shape)

# %%
# Remise à zéro des indexs
X_train_energ = X_train_energ.reset_index(drop=True)
X_test_energ = X_test_energ.reset_index(drop=True)
y_train_energ = y_train_energ.reset_index(drop=True)
y_test_energ = y_test_energ.reset_index(drop=True)

X_train_co2 = X_train_co2.reset_index(drop=True)
X_test_co2 = X_test_co2.reset_index(drop=True)
y_train_co2 = y_train_co2.reset_index(drop=True)
y_test_co2 = y_test_co2.reset_index(drop=True)

X_train_co2_q = X_train_co2_q.reset_index(drop=True)
X_test_co2_q = X_test_co2_q.reset_index(drop=True)
y_train_co2_q = y_train_co2_q.reset_index(drop=True)
y_test_co2_q = y_test_co2_q.reset_index(drop=True)

# %%
# Encodage des variables catégorielles
X_train_energ = pd.get_dummies(X_train_energ, drop_first=True)
X_train_co2 = pd.get_dummies(X_train_co2, drop_first=True)
X_train_co2_q = pd.get_dummies(X_train_co2_q, drop_first=True)

X_test_energ = pd.get_dummies(X_test_energ, drop_first=True)
X_test_co2 = pd.get_dummies(X_test_co2, drop_first=True)
X_test_co2_q = pd.get_dummies(X_test_co2_q, drop_first=True)

# %%
# Vérification normalité des données train
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(X_train_energ['puiss_admin_98'], line='45', fit = True)
plt.title('puiss_admin_98')
plt.show()


sm.qqplot(X_train_energ['puiss_max'], line='45', fit = True)
plt.title('puiss_max')
plt.show()


sm.qqplot(X_train_energ['co_typ_1'], line='45', fit = True)
plt.title('co_typ_1')
plt.show()


sm.qqplot(X_train_energ['hcnox'], line='45', fit = True)
plt.title('hcnox')
plt.show()


sm.qqplot(X_train_energ['ptcl'], line='45', fit = True)
plt.title('ptcl')
plt.show()


sm.qqplot(X_train_energ['masse_ordma_min'], line='45', fit = True)
plt.title('masse_ordma_min')
plt.show()


sm.qqplot(X_train_energ['masse_ordma_max'], line='45', fit = True)
plt.title('masse_ordma_max')
plt.show()


# %%
# Vérification normalité des données test
sm.qqplot(X_test_energ['puiss_admin_98'], line='45', fit = True)
plt.title('puiss_admin_98')
plt.show()


sm.qqplot(X_test_energ['puiss_max'], line='45', fit = True)
plt.title('puiss_max')
plt.show()


sm.qqplot(X_test_energ['co_typ_1'], line='45', fit = True)
plt.title('co_typ_1')
plt.show()


sm.qqplot(X_test_energ['hcnox'], line='45', fit = True)
plt.title('hcnox')
plt.show()


sm.qqplot(X_test_energ['ptcl'], line='45', fit = True)
plt.title('ptcl')
plt.show()


sm.qqplot(X_test_energ['masse_ordma_min'], line='45', fit = True)
plt.title('masse_ordma_min')
plt.show()


sm.qqplot(X_test_energ['masse_ordma_max'], line='45', fit = True)
plt.title('masse_ordma_max')
plt.show()

# %%
# Feature scaling
# Aucune donnée ne suit une loi normale --> normalisation

scaler = MinMaxScaler()

num = ['puiss_admin_98', 'puiss_max', 'co_typ_1', 'hcnox', 'ptcl', 'masse_ordma_min','masse_ordma_max']
X_train_energ.loc[:,num] = scaler.fit_transform(X_train_energ[num])
X_test_energ.loc[:,num] = scaler.transform(X_test_energ[num])

X_train_co2.loc[:,num] = scaler.fit_transform(X_train_co2[num])
X_test_co2.loc[:,num] = scaler.transform(X_test_co2[num])

X_train_co2_q.loc[:,num] = scaler.fit_transform(X_train_co2_q[num])
X_test_co2_q.loc[:,num] = scaler.transform(X_test_co2_q[num])

# %%
# Modèles "co2"

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


cl1_co2 = LinearRegression()
cl1_co2.fit(X_train_co2, y_train_co2)
print("Accuracy score de la Régression Linéaire : ",
      cl1_co2.score(X_test_co2, y_test_co2))

cl2_co2 = DecisionTreeRegressor(random_state = 42, max_depth = 5)
cl2_co2.fit(X_train_co2, y_train_co2)
print("Accuracy score de l'Arbre de Régression : ",
      cl2_co2.score(X_test_co2, y_test_co2))

cl3_co2 = RandomForestRegressor(random_state = 42, max_depth = 5)
cl3_co2.fit(X_train_co2, y_train_co2)
print("Accuracy score du Random Forest : ",
      cl3_co2.score(X_test_co2, y_test_co2))

# %%
# Visualisation des données
# Modèle linéaire
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize = (10,10))
pred_test = cl1_co2.predict(X_test_co2)
plt.scatter(pred_test, y_test_co2, c='green')

plt.plot((y_test_co2.min(), y_test_co2.max()), (y_test_co2.min(), y_test_co2.max()), color = 'red')
plt.xlabel("prediction")
plt.ylabel("vrai valeur")
plt.title('Régression Linéaire pour la prédiction des émissions de CO2 des voitures')

plt.show();

# %%
# Visualisation des données
# Arbre de régression 

from sklearn.tree import plot_tree
plt.figure(figsize = (270,30))
plot_tree(cl2_co2, 
          feature_names = X_train_co2.columns,
          fontsize=35,
          filled = True, 
          rounded = True)

plt.show()

# %%
# Visualisation des données
# Random Forest
from sklearn import tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (270,30), dpi=200)
tree.plot_tree(cl3_co2.estimators_[0],
               feature_names = X_train_co2.columns,
               fontsize=35, 
               filled = True);
fig.savefig('cl3_co2_individualtree.png')

# %%
# Calcul du score
print("score train : " , cl1_co2.score(X_train_co2, y_train_co2).round(4))
print("score test : ", cl1_co2.score(X_test_co2,y_test_co2).round(4))

print("score train : " , cl2_co2.score(X_train_co2, y_train_co2).round(4))
print("score test : ", cl2_co2.score(X_test_co2,y_test_co2).round(4))

print("score train : " , cl3_co2.score(X_train_co2, y_train_co2).round(4))
print("score test : ", cl3_co2.score(X_test_co2,y_test_co2).round(4))

# %%
# Calcul des métriques 
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y_pred_co2_1 = cl1_co2.predict(X_test_co2)
y_pred_co2_train_1 = cl1_co2.predict(X_train_co2)

y_pred_co2_2 = cl2_co2.predict(X_test_co2)
y_pred_co2_train_2 = cl2_co2.predict(X_train_co2)

y_pred_co2_3 = cl3_co2.predict(X_test_co2)
y_pred_co2_train_3 = cl3_co2.predict(X_train_co2)

# jeu d'entraînement 
mae_reglinear_train = mean_absolute_error(y_train_co2,y_pred_co2_train_1).round(2)
mse_reglinear_train = mean_squared_error(y_train_co2,y_pred_co2_train_1,squared=True).round(2)
rmse_reglinear_train = mean_squared_error(y_train_co2,y_pred_co2_train_1,squared=False).round(2)

mae_TreeReg_train = mean_absolute_error(y_train_co2,y_pred_co2_train_2).round(2)
mse_TreeReg_train = mean_squared_error(y_train_co2,y_pred_co2_train_2,squared=True).round(2)
rmse_TreeReg_train = mean_squared_error(y_train_co2,y_pred_co2_train_2,squared=False).round(2)

mae_random_forest_train = mean_absolute_error(y_train_co2,y_pred_co2_train_3).round(2)
mse_random_forest_train = mean_squared_error(y_train_co2,y_pred_co2_train_3,squared=True).round(2)
rmse_random_forest_train = mean_squared_error(y_train_co2,y_pred_co2_train_3,squared=False).round(2)

# jeu de test 
mae_reglinear_test = mean_absolute_error(y_test_co2,y_pred_co2_1).round(2)
mse_reglinear_test = mean_squared_error(y_test_co2,y_pred_co2_1,squared=True).round(2)
rmse_reglinear_test = mean_squared_error(y_test_co2,y_pred_co2_1,squared=False).round(2)

mae_TreeReg_test = mean_absolute_error(y_test_co2,y_pred_co2_2).round(2)
mse_TreeReg_test = mean_squared_error(y_test_co2,y_pred_co2_2,squared=True).round(2)
rmse_TreeReg_test = mean_squared_error(y_test_co2,y_pred_co2_2,squared=False).round(2)

mae_random_forest_test = mean_absolute_error(y_test_co2,y_pred_co2_3).round(2)
mse_random_forest_test = mean_squared_error(y_test_co2,y_pred_co2_3,squared=True).round(2)
rmse_random_forest_test = mean_squared_error(y_test_co2,y_pred_co2_3,squared=False).round(2)

# Creation d'un dataframe pour comparer les metriques des deux algorithmes 
data = {'MAE train': [mae_reglinear_train, mae_TreeReg_train, mae_random_forest_train],
        'MAE test': [mae_reglinear_test, mae_TreeReg_test, mae_random_forest_test],
        'MSE train': [mse_reglinear_train, mse_TreeReg_train,  mse_random_forest_train],
        'MSE test': [mse_reglinear_test, mse_TreeReg_test, mse_random_forest_test],
        'RMSE train': [rmse_reglinear_train, rmse_TreeReg_train, rmse_random_forest_train],
        'RMSE test': [rmse_reglinear_test, rmse_TreeReg_test, rmse_random_forest_test]}
  
# Creer DataFrame
df = pd.DataFrame(data, index = ['Linear Regression', 'Regression Tree', 'Random Forest '])

df.head()


# %%
# Visualisation feature importance meilleur modèle
import matplotlib.pyplot as plt

feat_importances = pd.DataFrame(cl3_co2.feature_importances_, index = X_train_co2.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(8,6))
plt.show()

# %%
# Visualisation valeurs shapley du meilleur modèle
import shap

explainer_randomforrest_co2 = shap.TreeExplainer(cl3_co2)
shap_values = explainer_randomforrest_co2.shap_values(X_train_co2)

shap.summary_plot(shap_values, X_train_co2) 
# Graphique résumant l'impact des variables sur le resultat de la prédiction pour chaque modalité de CO2


# %%
# Modèles Random Forest avec moins de varibales
X_train_reduced_co2 = X_train_co2[['masse_ordma_max','typ_boite_nb_rapp_M','hcnox']]
X_test_reduced_co2 = X_test_co2[['masse_ordma_max','typ_boite_nb_rapp_M','hcnox']]

cl3_co2_reduced = cl3_co2.fit(X_train_reduced_co2 , y_train_co2)

print(cl3_co2_reduced.score(X_train_reduced_co2, y_train_co2).round(4))
print(cl3_co2_reduced.score(X_test_reduced_co2, y_test_co2).round(4))


y_pred_co2_reduced = cl3_co2.predict(X_test_reduced_co2)
y_pred_co2_train_reduced = cl3_co2.predict(X_train_reduced_co2)

# jeu d'entraînement 
mae_random_forest_train = mean_absolute_error(y_train_co2,y_pred_co2_train_reduced).round(2)
mse_random_forest_train = mean_squared_error(y_train_co2,y_pred_co2_train_reduced,squared=True).round(2)
rmse_random_forest_train = mean_squared_error(y_train_co2,y_pred_co2_train_reduced,squared=False).round(2)

# jeu de test 
mae_random_forest_test = mean_absolute_error(y_test_co2,y_pred_co2_reduced).round(2)
mse_random_forest_test = mean_squared_error(y_test_co2,y_pred_co2_reduced,squared=True).round(2)
rmse_random_forest_test = mean_squared_error(y_test_co2,y_pred_co2_reduced,squared=False).round(2)

# Creation d'un dataframe pour comparer les metriques des deux algorithmes 
data = {'MAE train': [mae_random_forest_train],
        'MAE test': [mae_random_forest_test],
        'MSE train': [mse_random_forest_train],
        'MSE test': [mse_random_forest_test],
        'RMSE train': [rmse_random_forest_train],
        'RMSE test': [rmse_random_forest_test]}
  
# Creer DataFrame
df = pd.DataFrame(data, index = ['Random Forest '])

df.head()


# Résultats moins bien que le full modèle

# %%
# Visualisation des données réelles et prédites
from sklearn import tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (270,30), dpi=200)
tree.plot_tree(cl3_co2_reduced.estimators_[0],
               feature_names = ['masse_ordma_max','typ_boite_nb_rapp_M','hcnox'], 
               fontsize = 35,
               filled = True);
fig.savefig('cl3_co2_reduced_individualtree.png')

# %%
# Modèles "étiquette énergétique"
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


cl1_energ = LogisticRegression(random_state = 42)
cl1_energ.fit(X_train_energ, y_train_energ)
print("Accuracy score de la Régression Logistique : ",
      cl1_energ.score(X_test_energ, y_test_energ))

cl2_energ = DecisionTreeClassifier(random_state = 42, max_depth = 5)
cl2_energ.fit(X_train_energ, y_train_energ)
print("Accuracy score de l'Arbre de Décision : ",
      cl2_energ.score(X_test_energ, y_test_energ))

cl3_energ = RandomForestClassifier(random_state = 42, max_depth = 5)
cl3_energ.fit(X_train_energ, y_train_energ)
print("Accuracy score du Random Forest : ",
      cl3_energ.score(X_test_energ, y_test_energ))

# %%
# Visualisation des données
# Arbre de décision 

plt.figure(figsize = (270,30))
plot_tree(cl2_energ, 
          feature_names = X_train_energ.columns,
          fontsize=35,
          filled = True, 
          rounded = True)

plt.show()

# %%
# Visualisation des données
# Random Forest

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (270,30), dpi=200)
tree.plot_tree(cl3_energ.estimators_[0],
               feature_names = X_train_energ.columns,
               fontsize=35,
               filled = True);
fig.savefig('cl3_energ_individualtree.png')

# %%
# Calcul de l'accurancy

print("score train : " , cl1_energ.score(X_train_energ, y_train_energ).round(4))
print("score test : ", cl1_energ.score(X_test_energ,y_test_energ).round(4))

print("score train : " , cl2_energ.score(X_train_energ, y_train_energ).round(4))
print("score test : ", cl2_energ.score(X_test_energ,y_test_energ).round(4))

print("score train : " , cl3_energ.score(X_train_energ, y_train_energ).round(4))
print("score test : ", cl3_energ.score(X_test_energ,y_test_energ).round(4))

# %%
# calcul des métriques

from sklearn.metrics import classification_report

y_pred_energ_log = cl1_energ.predict(X_test_energ)
y_pred_energ_tree = cl2_energ.predict(X_test_energ)
y_pred_energ_rd_tree = cl3_energ.predict(X_test_energ)

display(pd.crosstab(y_test_energ,y_pred_energ_log, rownames=['Realité'], colnames=['Prédiction']))
display(pd.crosstab(y_test_energ,y_pred_energ_tree, rownames=['Realité'], colnames=['Prédiction']))
display(pd.crosstab(y_test_energ,y_pred_energ_rd_tree, rownames=['Realité'], colnames=['Prédiction']))

print(classification_report(y_test_energ, y_pred_energ_log, ))
print(classification_report(y_test_energ, y_pred_energ_tree, ))
print(classification_report(y_test_energ, y_pred_energ_rd_tree, ))


# %%
# Visualisation feature importance meilleur modèle
feat_importances = pd.DataFrame(cl2_energ.feature_importances_, index = X_train_energ.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(8,6))
plt.show()

# %%
# Visualisation valeurs shapley du meilleur modèle
import shap
explainer = shap.TreeExplainer(cl2_energ)
shap_values = explainer.shap_values(X_train_energ)

shap.summary_plot(shap_values, X_train_energ) 
# Graphique résumant l'impact des variables sur le résultat de la prédiction pour chaque classe énergétique 


# %%
# Graphiques résumant l'imapct des variables classe par classe 
shap.summary_plot(shap_values[0], X_train_energ) # Classe energetique G
shap.summary_plot(shap_values[1], X_train_energ) # Classe energetique F
shap.summary_plot(shap_values[2], X_train_energ) # Classe energetique E
shap.summary_plot(shap_values[3], X_train_energ) # Classe energetique D
shap.summary_plot(shap_values[4], X_train_energ) # Classe energetique C
shap.summary_plot(shap_values[5], X_train_energ) # Classe energetique B
shap.summary_plot(shap_values[6], X_train_energ) # Classe energetique A

# %%
# Test amélioration du modèle avec variables > 10%
X_train_reduced_energ = X_train_energ[['typ_boite_nb_rapp_M','carrosserie_MINIBUS', 'co_typ_1', 'masse_ordma_max']]
X_test_reduced_energ = X_test_energ[['typ_boite_nb_rapp_M','carrosserie_MINIBUS', 'co_typ_1', 'masse_ordma_max']]
cl2_energ_reduced = cl2_energ.fit(X_train_reduced_energ , y_train_energ)

print(cl2_energ_reduced.score(X_train_reduced_energ, y_train_energ))
print(cl2_energ_reduced.score(X_test_reduced_energ, y_test_energ))


# %%
feat_importances = pd.DataFrame(cl2_energ_reduced.feature_importances_, index = X_train_reduced_energ.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(8,6))
plt.show()

# %%
# Modèles "CO2 quantile"

cl1_co2_q = LogisticRegression()
cl1_co2_q.fit(X_train_co2_q, y_train_co2_q)
print("Accuracy score de la Régression Logistique : ",
      cl1_co2_q.score(X_test_co2_q, y_test_co2_q))

cl2_co2_q = DecisionTreeClassifier(random_state = 42, max_depth = 5)
cl2_co2_q.fit(X_train_co2_q, y_train_co2_q)
print("Accuracy score de l'Arbre de Décision : ",
      cl2_co2_q.score(X_test_co2_q, y_test_co2_q))

cl3_co2_q = RandomForestClassifier(random_state = 42, max_depth = 5)
cl3_co2_q.fit(X_train_co2_q, y_train_co2_q)
print("Accuracy score du Random Forest : ",
      cl3_co2_q.score(X_test_co2_q, y_test_co2_q))

# %%
# Visualisation des données
# Arbre de décision 

plt.figure(figsize = (270,30))
plot_tree(cl2_co2_q, 
          feature_names = X_train_co2_q.columns,
          fontsize=35,
          filled = True, 
          rounded = True)

plt.show()

# %%
# Visualisation des données
# Random Forest

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (270,30), dpi=200)
tree.plot_tree(cl3_co2_q.estimators_[0],
               feature_names = X_train_co2_q.columns,
               fontsize=35,
               filled = True);
fig.savefig('cl3_co2_q_individualtree.png')

# %%
# Calcul de l'accurancy

print("score train : " , cl1_co2_q.score(X_train_co2_q, y_train_co2_q).round(4))
print("score test : ", cl1_co2_q.score(X_test_co2_q,y_test_co2_q).round(4))

print("score train : " , cl2_co2_q.score(X_train_co2_q, y_train_co2_q).round(4))
print("score test : ", cl2_co2_q.score(X_test_co2_q,y_test_co2_q).round(4))

print("score train : " , cl3_co2_q.score(X_train_co2_q, y_train_co2_q).round(4))
print("score test : ", cl3_co2_q.score(X_test_co2_q,y_test_co2_q).round(4))

# %%
# calcul des métriques

from sklearn.metrics import classification_report

y_pred_co2_q_log = cl1_co2_q.predict(X_test_co2_q)
y_pred_co2_q_tree = cl2_co2_q.predict(X_test_co2_q)
y_pred_co2_q_rd_tree = cl3_co2_q.predict(X_test_co2_q)

display(pd.crosstab(y_test_co2_q,y_pred_co2_q_log, rownames=['Realité'], colnames=['Prédiction']))
display(pd.crosstab(y_test_co2_q,y_pred_co2_q_tree, rownames=['Realité'], colnames=['Prédiction']))
display(pd.crosstab(y_test_co2_q,y_pred_co2_q_rd_tree, rownames=['Realité'], colnames=['Prédiction']))

print(classification_report(y_test_co2_q, y_pred_co2_q_log, ))
print(classification_report(y_test_co2_q, y_pred_co2_q_tree, ))
print(classification_report(y_test_co2_q, y_pred_co2_q_rd_tree, ))


# Conclusion Modèles "CO2 quantile" = résultats moins probants que les deux autres modèles (co2 et étiquettes énergétiques )


