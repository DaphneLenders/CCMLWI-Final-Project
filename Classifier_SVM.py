from sklearn import svm

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(feature_matrix_train, train_labels)
    validation_prediction = svm_classifier.predict(feature_matrix_validation)
    return validation_prediction

