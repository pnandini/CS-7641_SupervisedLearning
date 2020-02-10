
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
import os

print("Begining Result process")

# This file moves the files along their required processes
folder = "data_final"
data_sets = ["anime.csv", "heart.csv"]
results = {'anime':{}, 'heart':{}}

## Process Anime
print("Processing Anime...")
train_loc = os.path.join(folder, "train_"+data_sets[0])
train_df = pd.read_csv(train_loc)
test_loc = os.path.join(folder, "test_"+data_sets[0])
test_df = pd.read_csv(test_loc)

# Features are everything but the last column
X = train_df.iloc[:,:-1]
X_test = test_df.iloc[:,:-1]

#Label is the last column
Y = train_df.iloc[:,-1]
Y_test = test_df.iloc[:,-1]

# Decission Tree
start_time = time.time()
estimator = DecisionTreeClassifier(criterion='gini', max_depth=21, min_samples_leaf=2)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['anime']['DT'] = {'score': score, 'time': elapsed_time}

# Boosted DT
start_time = time.time()
estimator = GradientBoostingClassifier(max_depth=5, min_samples_leaf=3, n_estimators=150)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['anime']['Boost'] = {'score': score, 'time': elapsed_time}

# KNN
start_time = time.time()
estimator = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=22,weights='distance')
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['anime']['KNN'] = {'score': score, 'time': elapsed_time}

#SVM
start_time = time.time()
estimator = SVC(kernel='rbf',degree=2,gamma='scale',random_state=0)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['anime']['SVM'] = {'score': score, 'time': elapsed_time}

#Neural
start_time = time.time()
estimator = MLPClassifier(activation='logistic',alpha=0.0001,beta_2=0.7,max_iter=500,random_state=0)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['anime']['Neural'] = {'score': score, 'time': elapsed_time}


## Process Anime
print("Processing Heart...")
train_loc = os.path.join(folder, "train_"+data_sets[1])
train_df = pd.read_csv(train_loc)
test_loc = os.path.join(folder, "test_"+data_sets[1])
test_df = pd.read_csv(test_loc)

# Features are everything but the last column
X = train_df.iloc[:,:-1]
X_test = test_df.iloc[:,:-1]

#Label is the last column
Y = train_df.iloc[:,-1]
Y_test = test_df.iloc[:,-1]

# Decission Tree
start_time = time.time()
estimator = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_leaf=3)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['heart']['DT'] = {'score': score, 'time': elapsed_time}

# Boosted DT
start_time = time.time()
estimator = GradientBoostingClassifier(max_depth=1, min_samples_leaf=1, n_estimators=100)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['heart']['Boost'] = {'score': score, 'time': elapsed_time}

# KNN
start_time = time.time()
estimator = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=11,weights='uniform')
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['heart']['KNN'] = {'score': score, 'time': elapsed_time}

#SVM
start_time = time.time()
estimator = SVC(C=3.0,kernel='rbf',degree=3,gamma='scale',random_state=0)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['heart']['SVM'] = {'score': score, 'time': elapsed_time}

#Neural
start_time = time.time()
estimator = MLPClassifier(activation='logistic',alpha=0.01,beta_2=0.7,max_iter=1000,random_state=0)
estimator.fit(X, Y)
score = estimator.score(X_test, Y_test)
elapsed_time = (time.time() - start_time)
results['heart']['Neural'] = {'score': score, 'time': elapsed_time}


#Print the results
print(results)