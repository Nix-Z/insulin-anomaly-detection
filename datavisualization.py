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
    
    percentile_25 = data[feature].quantile(0.25)
    percentile_75 = data[feature].quantile(0.75)
    iqr = percentile_75 - percentile_25
    upper_limit = percentile_75 + 1.5 * iqr
    lower_limit = percentile_25 - 1.5 * iqr

    data['Check_Outliers'] = np.nan

    for index in range(len(data)): #Wrong but on the right path?
        if lower_limit > data.loc[index, feature] or upper_limit < data.loc[index, feature]:
            data.loc[index, 'Check_Outliers'] = -1
        else:
            data.loc[index, 'Check_Outliers'] = 1
    
    data['Check_Outliers'] = data['Check_Outliers'].astype(int)
    print(data['Check_Outliers'].value_counts().sort_values(ascending=False))
    data.replace({'Check_Outliers': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph IQR Method results
    fig_2 = px.scatter(data, y=data[feature], color="Check_Outliers", title="Outliers Detected via IQR Method")
    fig_2.update_xaxes(showgrid=False)
    fig_2.update_yaxes(showgrid=False)
    fig_2.show()
    
    X = data[feature].values
    
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

    # Anamoly detection via Robust Covariance
    model = EllipticEnvelope(random_state=1)
    predictions = model.fit_predict(X.reshape(-1,1))
    data['Status'] = predictions
    print(data['Status'].value_counts().sort_values(ascending=False))

    data.replace({'Status': {-1:'outlier', 1:'inlier'}}, inplace=True)

    # Graph Robust Covariance results
    fig_7 = px.scatter(data, y=data[feature], color="Status", title="Outliers Detected via Robust Covariance Algorithm")
    fig_7.update_xaxes(showgrid=False)
    fig_7.update_yaxes(showgrid=False)
    fig_7.show()
    
    return data

visualize_data()
