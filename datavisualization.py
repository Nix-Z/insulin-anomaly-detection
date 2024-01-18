import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.covariance import EllipticEnvelope
from data_analysis import analyze_data

def visualize_data():
    data = analyze_data()

    # Create correlation plot for features
    corr_data_num = data.corr()
    print(corr_data_num)
    fig_1 = px.imshow(corr_data_num, labels=dict(color='Correlation'), x=corr_data_num.columns, y=corr_data_num.index, text_auto=True)
    fig_1.show()

    feature = 'Insulin'
    X = data[feature].values
    
    # Anamoly detection via Robust Covariance
    model = EllipticEnvelope(random_state=1)
    predictions = model.fit_predict(X.reshape(-1,1))
    data['Status'] = predictions
    print(data['Status'].value_counts().sort_values(ascending=False))

    data.replace({'Status': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph Robust Covariance results
    fig_2 = px.scatter(data, y=data[feature], color="Status", title="Outliers Detected via Robust Covariance Algorithm")
    fig_2.update_xaxes(showgrid=False)
    fig_2.update_yaxes(showgrid=False)
    fig_2.show()
    
    # Anamoly detection via IsolationForest
    model = IsolationForest(random_state=1)
    predictions = model.fit_predict(X.reshape(-1,1))
    data['Status'] = predictions
    print(data['Status'].value_counts().sort_values(ascending=False))

    data.replace({'Status': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph IsolationForest results
    fig_3 = px.scatter(data, y=data[feature], color="Status", title="Outliers Detected via IsolationForest Algorithm")
    fig_3.update_xaxes(showgrid=False)
    fig_3.update_yaxes(showgrid=False)
    fig_3.show()

    # Anamoly detection via Local Outlier Factor
    model = LocalOutlierFactor()
    predictions = model.fit_predict(X.reshape(-1,1))
    data['Status'] = predictions
    print(data['Status'].value_counts().sort_values(ascending=False))

    data.replace({'Status': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph Local Outlier Factor results
    fig_4 = px.scatter(data, y=data[feature], color="Status", title="Outliers Detected via Local Outlier Factor Algorithm")
    fig_4.update_xaxes(showgrid=False)
    fig_4.update_yaxes(showgrid=False)
    fig_4.show()

    # Anamoly detection via One Class SVM
    model = OneClassSVM(gamma='auto')
    predictions = model.fit_predict(X.reshape(-1,1))
    data['Status'] = predictions
    print(data['Status'].value_counts().sort_values(ascending=False))

    data.replace({'Status': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph One Class SVM results
    fig_5 = px.scatter(data, y=data[feature], color="Status", title="Outliers Detected via One Class SVM Algorithm")
    fig_5.update_xaxes(showgrid=False)
    fig_5.update_yaxes(showgrid=False)
    fig_5.show()

    # Anamoly detection via SGD One Class SVM
    model = SGDOneClassSVM(random_state=1)
    predictions = model.fit_predict(X.reshape(-1,1))
    data['Status'] = predictions
    print(data['Status'].value_counts().sort_values(ascending=False))

    data.replace({'Status': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph SGD One Class SVM results
    fig_6 = px.scatter(data, y=data[feature], color="Status", title="Outliers Detected via SGD One Class SVM Algorithm")
    fig_6.update_xaxes(showgrid=False)
    fig_6.update_yaxes(showgrid=False)
    fig_6.show()
    
    return data

visualize_data()
    
    
