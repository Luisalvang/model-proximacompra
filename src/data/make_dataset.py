
# Script de Preparación de Datos
###################################
import pandas as pd
import os
from datetime import datetime
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../../data/raw/', filename)) #.set_index('ID')
    print(filename, ' cargado correctamente')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# Realizamos la transformación de datos
def data_preparation(df):
    tx_data = df.copy()
    tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
    tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_6m = tx_uk[(tx_uk.InvoiceDate < datetime(2011,9,1)) & (tx_uk.InvoiceDate >= datetime(2011,3,1))].reset_index(drop=True)
    tx_next = tx_uk[(tx_uk.InvoiceDate >= datetime(2011,9,1)) & (tx_uk.InvoiceDate < datetime(2011,12,1))].reset_index(drop=True)
    tx_user = pd.DataFrame(tx_6m['CustomerID'].unique())
    tx_user.columns = ['CustomerID']
    
    # Agregando etiquetas
    tx_next_first_purchase = tx_next.groupby('CustomerID').InvoiceDate.min().reset_index()
    tx_next_first_purchase.columns = ['CustomerID','MinPurchaseDate']
    tx_last_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_last_purchase.columns = ['CustomerID','MaxPurchaseDate']
    tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='CustomerID',how='left')
    tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days
    tx_user = pd.merge(tx_user, tx_purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')
    tx_user = tx_user.fillna(999)
    
    #Recency
    tx_max_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
    tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

    # Frecuencia
    tx_frequency = tx_6m.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID','Frequency']
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
    tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

    # Monetary Value
    tx_6m['Revenue'] = tx_6m['UnitPrice'] * tx_6m['Quantity']
    tx_revenue = tx_6m.groupby('CustomerID').Revenue.sum().reset_index()
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
    tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

    # Overall Segmentation
    tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 

    # Agregando features
    #create a dataframe with CustomerID and Invoice Date
    tx_day_order = tx_6m[['CustomerID','InvoiceDate']]
    #Convert Invoice Datetime to day
    tx_day_order['InvoiceDay'] = tx_6m['InvoiceDate'].dt.date
    tx_day_order = tx_day_order.sort_values(['CustomerID','InvoiceDate'])
    #Drop duplicates
    tx_day_order = tx_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')
    #shifting last 3 purchase dates
    tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(1)
    tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(2)
    tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(3)
    tx_day_order['InvoiceDay'] = pd.to_datetime(tx_day_order['InvoiceDay'], errors='coerce')
    tx_day_order['PrevInvoiceDate'] = pd.to_datetime(tx_day_order['PrevInvoiceDate'], errors='coerce')
    tx_day_order['T2InvoiceDate'] = pd.to_datetime(tx_day_order['T2InvoiceDate'], errors='coerce')
    tx_day_order['T3InvoiceDate'] = pd.to_datetime(tx_day_order['T3InvoiceDate'], errors='coerce')
    tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
    tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days
    tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days
    tx_day_diff = tx_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
    tx_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']
    tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerID'],keep='last')
    tx_day_order_last = tx_day_order_last.dropna()
    tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='CustomerID')
    tx_user = pd.merge(tx_user, tx_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')

    # Grouping the label
    tx_class = tx_user.copy()
    tx_class = pd.get_dummies(tx_class)
    tx_class['NextPurchaseDayRange'] = 2
    tx_class.loc[tx_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
    tx_class.loc[tx_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0
    tx_class = tx_class.drop('NextPurchaseDay',axis=1)

    return tx_class


# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('data.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'DayDiff', 'DayDiff2', 'DayDiff3', 'DayDiffMean', 'DayDiffStd',
       'Segment_High-Value', 'Segment_Low-Value', 'Segment_Mid-Value',
       'NextPurchaseDayRange'],'data_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('data_val.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'DayDiff', 'DayDiff2', 'DayDiff3', 'DayDiffMean', 'DayDiffStd',
       'Segment_High-Value', 'Segment_Low-Value', 'Segment_Mid-Value',
       'NextPurchaseDayRange'],'data_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('data_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['CustomerID', 'Recency', 'RecencyCluster', 'Frequency',
       'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
       'DayDiff', 'DayDiff2', 'DayDiff3', 'DayDiffMean', 'DayDiffStd',
       'Segment_High-Value', 'Segment_Low-Value', 'Segment_Mid-Value'],
       'data_score.csv')


if __name__ == "__main__":
    main()