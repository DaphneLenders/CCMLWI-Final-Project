from sklearn.ensemble import RandomForestClassifier

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    rf = RandomForestClassifier(random_state=1)
    rf.fit(feature_matrix_train, train_labels)
    validation_prediction = rf.predict(feature_matrix_validation)
    return validation_prediction