# Data manipulation
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs
from matplotlib import pyplot as plt

# Learning and preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#data analysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import itertools
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
#CWD = path.dirname(path.realpath(__file__))
"""
It must contain files with raw data
"""
# DATA_PATH = path.join(CWD, 'data')
DATA_PATH = path.abspath("src/data/")
TEST_DATA_PATH = path.join(DATA_PATH, 'test')
TRAIN_DATA_PATH = path.join(DATA_PATH, 'train')
VALIDATION_DATA_PATH = path.join(DATA_PATH, 'validation')
DATA_FILE = path.join(DATA_PATH, 'data.csv')


def preprocessing_data(data):
    modifedData=data.fillna(' ')
    modifedData.to_csv('modifiedData.csv',index=False)
    modifedData['Position'] = modifedData['Position'].apply(
        lambda x: 1 if x.endswith('B') else 0)
    y = modifedData['Position']
    x = modifedData.drop('Position', axis = 1)
    x = x.astype(str).apply(preprocessing.LabelEncoder().fit_transform)
    return x,y

def accuracy_control(x,y):
    train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), 
                                                        x, 
                                                        y, 
                                                        cv=10,
                                                        scoring='accuracy',
                                                        n_jobs=-1, 
                                                        verbose = 1,
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def confusion_matrix_control(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ['No defender', 'Defender']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def precision_recall_curve_control(logreg, X_test,y_test):
    y_score = logreg.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
    plt.show()

def guality_control(x,y, X_test, y_test, predictions, logreg):
    accuracy_control(x,y)
    confusion_matrix_control(y_test, predictions) 
    precision_recall_curve_control(logreg, X_test, y_test)

    print(f1_score(y_test, predictions, average='macro'))  
    print(f1_score(y_test, predictions, average='micro'))  
    print(f1_score(y_test, predictions, average='weighted'))  
    print(f1_score(y_test, predictions, average=None))



def main():
    data = pd.read_csv(DATA_FILE)
    x, y = preprocessing_data(data)
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
    logreg = LogisticRegression(verbose=1)
    logreg.fit(X_train, y_train)
    pred_class = logreg.predict(X_test)
    accuracy_score(y_test, pred_class)
    X_shuf, Y_shuf = shuffle(x, y)
    guality_control(X_shuf, Y_shuf, X_test, y_test, pred_class, logreg)

if __name__ == '__main__':
    main()