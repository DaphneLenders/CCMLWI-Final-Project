import pandas as pd
from ExtractFeatures import extract_features
from sklearn.model_selection import KFold
from sklearn import svm, cross_validation, naive_bayes
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import xlsxwriter


def read_data():
    data = pd.read_excel('translated_texts.xlsx')
    data_without_neutrals = data[data.label != 'NEUTRAAL']
    data = data_without_neutrals.reset_index()
    return data



def eight_labels():
    data = read_data()
    folds = create_folds(data.text, data.label)
    return folds


def four_labels():
    data = read_data()
    data.label = ['SAMEN-BOVEN' if (x=='LEIDEND' or x== 'HELPEND') else x for x in data.label ]
    data.label = ['SAMEN-ONDER' if (x=='MEEWERKEND' or x == 'VOLGEND') else x for x in data.label]
    data.label = ['TEGEN-ONDER' if (x=='TERUGGETROKKEN' or x == 'OPSTANDIG') else x for x in data.label]
    data.label = ['TEGEN-BOVEN' if (x=='AANVALLEND'or x == 'COMPETITIEF') else x for x in data.label]
    folds = create_folds(data.text, data.label)
    return folds


def create_folds(train_data, labels):
    splitted_indeces = cross_validation.KFold(len(labels), n_folds=10, shuffle=True, random_state=1)
    folds = []
    for train_index, validation_index in splitted_indeces:
        train_data_new, validation_data_new = train_data.reindex(train_index), train_data.reindex(validation_index)
        train_labels_new, validation_labels_new = labels.reindex(train_index), labels.reindex(validation_index)
        folds.append((train_data_new, train_labels_new, validation_data_new, validation_labels_new))
    return folds


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


def writeInExcelSheet(rightPredictions, wrongPredictions):
    workbook = xlsxwriter.Workbook('Predictions.xlsx')
    rights = workbook.add_worksheet('Right Predictions')
    wrongs = workbook.add_worksheet('Wrong Predictions')

    rights.write(0, 0, 'Text')
    rights.write(0, 1, 'Label')
    rights.write(0, 2, 'Prediction')
    row = 1
    col = 0

    for text, true_label, prediction in rightPredictions:
        rights.write(row, col, text)
        rights.write(row, col+1, true_label)
        rights.write(row, col+2, prediction)
        col = 0
        row+=1

    wrongs.write(0, 0, 'Text')
    wrongs.write(0, 1, 'Label')
    wrongs.write(0, 2, 'Prediction')
    row = 1
    col = 0
    for text, true_label, prediction in wrongPredictions:
        wrongs.write(row, col, text)
        wrongs.write(row, col + 1, true_label)
        wrongs.write(row, col + 2, prediction)
        col = 0
        row +=1

    workbook.close()


# The text that have been assigned the correct rose-of-leary quadrant/octanct and the texts that have been assigned to the wrong
# rose-of-leary quadrant/octant are stored into seperate arrays. Afterwards the function 'writeInExcelSheet' is called so the
# data can be visualized in an Excel-file.
def makeOverview(test_prediction, test_data, labels):
   iterator = 0
   rightPredictions = []
   wrongPredictions = []
   for prediction in test_prediction:
       if prediction == labels.iloc[iterator]:
           rightPredictions.append((test_data.iloc[iterator], labels.iloc[iterator], prediction))
       else:
           wrongPredictions.append((test_data.iloc[iterator], labels.iloc[iterator], prediction))
       iterator+=1

   writeInExcelSheet(rightPredictions, wrongPredictions)