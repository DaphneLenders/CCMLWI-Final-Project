from sklearn import svm
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    svm_ecoc_classifier = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    svm_ecoc_classifier.fit(feature_matrix_train, train_labels)
    validation_prediction = svm_ecoc_classifier.predict(feature_matrix_validation)
    return validation_prediction