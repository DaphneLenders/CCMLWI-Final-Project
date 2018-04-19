import pandas as pd
from ExtractFeatures import extract_features
from sklearn.model_selection import KFold
from sklearn import svm, cross_validation
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from InverseDocumentFrequentizer import idf_vectorizer


def read_data():
    data = pd.read_excel('translated_texts.xlsx')
    return data

def create_folds(train_data, labels):
    splitted_indeces = cross_validation.KFold(len(labels), n_folds=10, shuffle=True, random_state=1)
    folds = []
    print(train_data[1136])
    for train_index, validation_index in splitted_indeces:
        try:
            print(train_data[1136])
        except:
            print("not here")
        train_data, validation_data = train_data.reindex(train_index), train_data.reindex(validation_index)
        train_labels, validation_labels = labels.reindex(train_index), labels.reindex(validation_index)
        folds.append((train_data, train_labels, validation_data, validation_labels))
    return folds

def classify(folds):
    scores = []
    for (train_data, train_labels, validation_data, validation_labels) in folds:
        # The inverse document frequencies of the train data are extracted
        print(idf_vectorizer(train_data))



        # Features are extracted
        feature_matrix_train = list(map(extract_features, train_data))
        # Oversampling is used to balance out dataset
        feature_matrix_train, train_labels = SMOTE().fit_sample(feature_matrix_train, train_labels)


        feature_matrix_validation = list(map(extract_features, validation_data))


        clf = svm.SVC(kernel='linear')
        clf.fit(feature_matrix_train, train_labels)
        validation_prediction = clf.predict(feature_matrix_validation)
        print(confusion_matrix(validation_labels, validation_prediction,
                               labels=['NEUTRAAL', 'MEEWERKEND', 'VOLGEND', 'TERUGGETROKKEN', 'OPSTANDIG', 'AANVALLEND', 'COMPETITIEF', 'LEIDEND', 'HELPEND']))
        score = evaluate(validation_labels, validation_prediction)
        scores.append(score)
    average_scores(scores)

# Given the predicted class labels and the true class labels, this function prints out the recall, precision and f1-score
# of a classifier.
def evaluate(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = 2 * (precision * recall)/(precision + recall)
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score


def average_scores(scores):
    average_recall = (sum(r for r, p, f in scores))/len(scores)
    average_prediction = (sum (p for r, p, f in scores))/len(scores)
    average_f1 = (sum(f for r, p, f in scores))/len(scores)

    print("Average Recall: %f" % average_recall)
    print("Average Prediction: %f" % average_prediction)
    print("Average F1-score: %f" % average_f1)


def main():
    data = read_data()
    folds = create_folds(data.text, data.label)
    classify(folds)




if __name__ == '__main__':
    main()