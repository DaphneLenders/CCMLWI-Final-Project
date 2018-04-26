from sklearn import naive_bayes

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    nb = naive_bayes.GaussianNB()
    nb.fit(feature_matrix_train, train_labels)
    validation_prediction = nb.predict(feature_matrix_validation)
    return validation_prediction