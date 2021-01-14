import time
import pandas as pd
from pyarrow import csv, parquet
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


file_csv_gz = 'AAPL.OQ.csv.gz'
file_parquet = 'AAPL.OQ.parquet'
file_hdf = 'AAPL.OQ.h5'

# CSVGZ TO PANDAS
t0 = time.time()
allevents = pd.read_csv(file_csv_gz)
t1 = time.time()
read_csvgz_time = t1 - t0
# CSVGZ TO PYARROW TO PANDAS
t0 = time.time()
table_1 = csv.read_csv(file_csv_gz)
df = table_1.to_pandas()
t1 = time.time()
read_csvgz_to_pyarrow_time = t1 - t0
# PARQUET TO PANDAS
t0 = time.time()
df = pd.read_parquet(file_parquet)
t1 = time.time()
read_parquet_time = t1 - t0
# PARQUET TO PYARROW TO PANDAS
t0 = time.time()
table_2 = parquet.read_table(file_parquet)
df = table_2.to_pandas()
t1 = time.time()
read_parquet_to_pyarrow_time = t1 - t0
# HDF5 TO PANDAS
t0 = time.time()
df = pd.read_hdf(file_hdf)
t1 = time.time()
read_hdf5_time = t1 - t0

# Feature engineering
t0 = time.time()
SimData = pd.DataFrame(index=allevents.index)
SimData['total-traded'] = allevents['trade_price'] * allevents['trade_volume']
SimData['bid-ask-spread'] = allevents['bid-price'] - allevents['ask-price']
SimData['imbalance'] = allevents['bid-volume'].rolling(5).sum() - allevents['ask-volume'].rolling(5).sum()
SimData['change'] = (allevents.trade_price - allevents.trade_price.shift(-30) > 0).astype('int')
SimData.dropna(how='any', inplace=True)

array = SimData.values
X = array[:, 0:3]
y = array[:, 3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=None)

# Use a Decision Tree Classifier for prediction
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
t1 = time.time()
prediction_time = t1 - t0

# Evaluates the predictions
# Accuracy
print("Accuracy of the prediction : \n" + str(accuracy_score(Y_validation, predictions)))

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, X_validation, Y_validation,
                                 display_labels=['0', '1'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

# Classification report
print("Classification report : ")
print(classification_report(Y_validation, predictions))
plt.show()

print('Reading time, csv.gz...', read_csvgz_time, 's')
print('Reading time, csv.gz via pyarrow...', read_csvgz_to_pyarrow_time, 's')
print('Reading time, parquet...', read_parquet_time, 's')
print('Reading time, parquet via pyarrow...', read_parquet_to_pyarrow_time, 's')
print('Reading time, hdf5...', read_hdf5_time, 's')
print('Prediction time...', prediction_time, 's')

