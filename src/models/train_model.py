# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################


import pandas as pd
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../../data/processed', filename)) #.set_index('ID')
    X_train = df.drop(['NextPurchaseDayRange'],axis=1)
    y_train = df[['NextPurchaseDayRange']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    # xgb_mod=xgb.XGBClassifier(max_depth=2, n_estimators=50, objective='binary:logistic', seed=0, silent=True, subsample=.8)
    # xgb_mod.fit(X_train, y_train
    xgb_mod = xgb.XGBClassifier().fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../../models/best_model.pkl'
    pickle.dump(xgb_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('data_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()