import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn.linear_model import RidgeClassifier

from tensorflow import keras
from tensorflow.keras import models

import matplotlib.pyplot as plt

min_max_scaler = preprocessing.MaxAbsScaler()

UNREST_COLUMNS = ["EVENT_ID_CNTY", 
                    "EVENT_DATE", 
                    "EVENT_TYPE", 
                    "REGION", 
                "FATALITIES",
                "TIMESTAMP"]

CASES_COLUMNS = ["iso_code",
                "continent",
                "location", 
                "date", 
                "total_cases", 
                "new_cases", 
                "total_deaths", 
                "reproduction_rate", 
                "hosp_patients", 
                "positive_rate", 
                "stringency_index", 
                "population",
                "median_age",
                "gdp_per_capita",
                "life_expectancy",
            ]

def serialize(dataFrame, column):
    return [x for x in range(len(dataFrame[column].unique()))]

def replaceDict(dataFrame, column):
    vals = serialize(dataFrame, column)
    return dict(zip(dataFrame[column].unique(), vals))

def multiSearch(df, column, searchTerms):
    if type(searchTerms) is list:
        return df.query(' | '.join(
            [f'{column} == "{term}"' for term in searchTerms]
        ))
    elif type(searchTerms) is str:
        return df.query(f'{column} == "{searchTerms}"')
    else:
        return df.query(f'{column} == {searchTerms}')

def multiContains(df, column, searchTerms):
    if type(searchTerms) is list:
        return df[df[column].str.contains('|'.join(searchTerms))]
    else:
        return df[df[column].str.contains(searchTerms)]

#Create the training data set of merged PD's and the result
def retrieveTrainingData(isoCodes):
    unrest_df = pd.read_csv("./coronavirus_Oct31.csv")
    unrest_df = unrest_df[unrest_df.EVENT_TYPE != 'Strategic developments']
 
    covid_cases_df = pd.read_csv("./owid-covid-data.csv")
    print(unrest_df.EVENT_TYPE.unique())

    unrest_df = unrest_df[unrest_df.columns.intersection(UNREST_COLUMNS)]
    covid_cases_df = covid_cases_df[covid_cases_df.columns.intersection(CASES_COLUMNS)]

    isoCodes = list(set([x[:3:1] for x in unrest_df['EVENT_ID_CNTY'].values]))
    #isoCodes =  isoCodes[:len(isoCodes)]

    unrest = multiContains(unrest_df, "EVENT_ID_CNTY", isoCodes)
    cases = multiSearch(covid_cases_df, 'iso_code', isoCodes)

    unrest.EVENT_DATE = pd.to_datetime(unrest.EVENT_DATE)
    cases.date = pd.to_datetime(cases.date) 

    merge = unrest.merge(cases, how="inner", left_on="EVENT_DATE", right_on="date")

    merge = merge.drop(['EVENT_ID_CNTY'], axis=1)
    merge = merge.drop_duplicates()

    issueType = merge['EVENT_TYPE']
    issueType = issueType.replace(replaceDict(unrest_df, "EVENT_TYPE"))


    merge = merge.drop(['EVENT_TYPE', 'EVENT_DATE', 'REGION', 'iso_code', 'continent', 'location', 'date', 'TIMESTAMP'], axis=1).fillna(0)

    return merge, issueType


merge2, issueType2 = retrieveTrainingData(['BOL'])

mergeNormal2 = min_max_scaler.fit_transform(merge2)
mergeNormal2 = pd.DataFrame(mergeNormal2)

merge, issueType = retrieveTrainingData(['AFG','ARG', 'BDG'])

mergeNormal = min_max_scaler.fit_transform(merge)
mergeNormal = np.array(mergeNormal)

X, y = retrieveTrainingData('AFG')
#clf = svm.SVC()
#print(issueType)
#clf.fit(mergeNormal, issueType)

#clf.predict(mergeNormal)

dTree = DecisionTreeClassifier(max_depth=5)
dTree.fit(mergeNormal, issueType)
pred = dTree.predict(mergeNormal)
pred2 = dTree.predict(mergeNormal2)
print("Decision tree")
print(precision_score(issueType, pred, average="micro"))
print(precision_score(issueType2, pred2, average="micro"))

ridge = RidgeClassifier()
ridge.fit(mergeNormal, issueType)
ridgePred = ridge.predict(mergeNormal)
ridgePredGen = ridge.predict(mergeNormal2)
print("Ridge")
print(precision_score(issueType, ridgePred, average="micro"))
print(precision_score(issueType2, ridgePredGen, average="micro"))

issueNN = []
for i in issueType:
  issueChild = []
  for j in range(0, 5):
    issueChild.append(0)
  issueChild[i] = 1
  issueNN.append(issueChild)

issueNN = np.array(issueNN)

model = keras.models.Sequential()


#model.add(keras.layers.Dense(12, activation='relu', input_shape=[12,]))
#model.add(keras.layers.Dense(12, activation='relu'))
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv1D(8, activation='relu', kernel_size=2, input_shape=(12,1)))
model.add(keras.layers.Conv1D(8, activation='relu', kernel_size=2))
model.add(keras.layers.MaxPool1D(pool_size=2))
model.add(keras.layers.Conv1D(16, activation='relu', kernel_size=3))
model.add(keras.layers.Conv1D(16, activation='relu', kernel_size=3))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(issueNN.shape[1], activation='softmax'))

opt = keras.optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),  metrics=["accuracy"])

model.summary()

batch_size = 128
epochs = 50

mergeNormal = mergeNormal.reshape(mergeNormal.shape[0], mergeNormal.shape[1], 1)

mergeNormalTrain, mergeNormalTest, issueTrain, issueTest = model_selection.train_test_split(mergeNormal, issueNN, test_size=0.33, random_state=12345)
history = model.fit(mergeNormalTrain, issueTrain, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1)

model.save("testUnrest.model")


plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.evaluate(mergeNormalTest,issueTest)

preds = model(mergeNormalTest, training=False)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(issueTest, axis=1)
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1, y_pred))