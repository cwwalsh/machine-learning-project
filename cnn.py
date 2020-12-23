import pandas as pd
import numpy as np
import math

from collections import Counter

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

from tensorflow import keras

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

def oneToOne(df, column):
    count = []
    ret = pd.DataFrame()
    
    count = [{"col": val, "count": df[df[column] == val].shape[0]} for val in df[column].unique()]
    count.sort(key=lambda x: x.get("count"))
    #print(df[df[column] ==  "Protests"].sample(count[0]))
    #print(count[0] * 10)
    i = 1
    for col in count:
        if ret.empty:
            ret = pd.DataFrame(df[df[column] == col.get("col")].sample(math.floor(count[0].get("count") * i), replace=True).drop_duplicates())
        else:
            ret = ret.append(df[df[column] == col.get("col")].sample(math.floor(count[0].get("count") * i), replace=True).drop_duplicates())
        i = i + 8
    return ret

def randomSample():
    #Read in data from files
    unrest_df = pd.read_csv("./coronavirus_Oct31.csv")
    #Remove rows with event type of strategic developments
    unrest_df = unrest_df[unrest_df.EVENT_TYPE != 'Strategic developments']
 
    covid_cases_df = pd.read_csv("./owid-covid-data.csv")

    unrest_df = unrest_df[unrest_df.columns.intersection(UNREST_COLUMNS)]
    covid_cases_df = covid_cases_df[covid_cases_df.columns.intersection(CASES_COLUMNS)]
    #Get data based on the input iso country codes
    #unrest = multiContains(unrest_df, "EVENT_ID_CNTY", isoCodes)
    #cases = multiSearch(covid_cases_df, 'iso_code', isoCodes)

    #Convert "date" type columns to dates
    unrest_df.EVENT_DATE = pd.to_datetime(unrest_df.EVENT_DATE)
    unrest_df.EVENT_ID_CNTY = unrest_df.EVENT_ID_CNTY.astype(str).str[:3]
    covid_cases_df.date = pd.to_datetime(covid_cases_df.date)
    #Merge the two datasets with an inner join on the date fields
    merge = unrest_df.merge(covid_cases_df, how="inner", left_on=["EVENT_DATE", 'EVENT_ID_CNTY'], right_on=["date", "iso_code"])
    #merge = merge.drop_duplicates(subset=['EVENT_DATE', 'EVENT_ID_CNTY'])
    #Drop the iso code to avoid duplicates
    
    merge = merge.drop(['EVENT_ID_CNTY'], axis=1)
    #Drop remaining duplicates
    #merge = merge.drop_duplicates()
    print(merge.shape)
    #Get the list of event types in this particular set of data
    #issueType = merge['EVENT_TYPE']
    #Serialize the data and return it as the expected values for training
    #issueType = issueType.replace(replaceDict(unrest_df, "EVENT_TYPE"))
    
    #result = oneToOne(merge, 'EVENT_TYPE')
    result = merge
    issueType = result['EVENT_TYPE']
    #Serialize the data and return it as the expected values for training
    issueType = issueType.replace(replaceDict(unrest_df, "EVENT_TYPE"))

    #Drop remaining unneeded data
    result = result.drop(['EVENT_TYPE', 'EVENT_DATE', 'REGION', 'iso_code', 'continent', 'location', 'date', 'TIMESTAMP'], axis=1).fillna(0)

    return result, issueType

min_max_scaler = preprocessing.MinMaxScaler()
merge, issueType = randomSample()
mergeNormal = min_max_scaler.fit_transform(merge)
mergeNormal = np.array(mergeNormal)

issueNN = []
for i in issueType:
  issueChild = []
  for j in range(0, 5):
    issueChild.append(0)
  issueChild[i] = 1
  issueNN.append(issueChild)

issueNN = np.array(issueNN)

overVal = max(Counter(issueType).values())
overKey = max(Counter(issueType), key=Counter(issueType).get)
overDict = {x: overVal for x in issueType}
over = SMOTE(sampling_strategy=overDict)

mergeNormal, issueNN = over.fit_resample(mergeNormal, issueNN)

mergeNormal = mergeNormal.reshape(mergeNormal.shape[0], mergeNormal.shape[1], 1)
mergeNormalTrain, mergeNormalTest, issueTrain, issueTest = model_selection.train_test_split(mergeNormal, issueNN, test_size=0.2)

model = keras.models.Sequential()

#Best: 83ish
model.add(keras.layers.Conv1D(12, activation='relu', kernel_size=3, padding='same', input_shape=(12,1)))
model.add(keras.layers.Conv1D(12, activation='relu', kernel_size=3, padding='same'))
model.add(keras.layers.MaxPool1D(strides=2))
model.add(keras.layers.Conv1D(32, activation='relu', kernel_size=5, padding='same'))
model.add(keras.layers.Conv1D(32, activation='relu', kernel_size=5, padding='same'))
model.add(keras.layers.Conv1D(32, activation='relu', kernel_size=7, padding='same'))
model.add(keras.layers.Conv1D(32, activation='relu', kernel_size=7, padding='same'))
model.add(keras.layers.LSTM(24))

model.add(keras.layers.Dense(issueNN.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01, decay=0.001),  metrics=["accuracy"])

model.summary()

batch_size = 128
epochs = 60
split = 0.1

history = model.fit(mergeNormalTrain, issueTrain, batch_size=batch_size,
                    epochs=epochs, validation_split=split)

model.save("unrest.model")

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