
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
import numpy as np

print("Begining DecissionTree process.")

# This file moves the files along their required processes
folder = "data_final"
data_sets = ["anime.csv", "heart.csv"]

## Process Anime
#print("Processing Anime...")
#train_anime_loc = os.path.join(folder, "train_"+data_sets[0])
#train_anime_df = pd.read_csv(train_anime_loc)
#test_anime_loc = os.path.join(folder, "test_"+data_sets[0])
#test_anime_df = pd.read_csv(test_anime_loc)
#
## Features are everything but the last column
#train_anime_X = train_anime_df.iloc[:,:-1]
#test_anime_X = test_anime_df.iloc[:,:-1]
#
##Label is the last column
#train_anime_Y = train_anime_df.iloc[:,-1]
#test_anime_Y = test_anime_df.iloc[:,-1]
#
## Train Decission Tree
#tree_para = {'max_depth':(np.arange(10)+1).tolist(), 'min_samples_leaf':[1,2,3,4], 'n_estimators':[50,100,150,200], 'random_state':[0]}
#dectree_anime = GridSearchCV(GradientBoostingClassifier(), tree_para, cv=5)
#dectree_anime.fit(train_anime_X, train_anime_Y)
#
#score = dectree_anime.score(test_anime_X, test_anime_Y)
#print("Original GridSearch Anime Test Score: " + str(score))
#print("Original Gridsearch Anime Params: " + str(dectree_anime.best_params_))


# Process Heart
print("Processing Heart...")
train_heart_loc = os.path.join(folder, "train_"+data_sets[1])
train_heart_df = pd.read_csv(train_heart_loc)
test_heart_loc = os.path.join(folder, "test_"+data_sets[1])
test_heart_df = pd.read_csv(test_heart_loc)

# Features are everything but the last column
train_heart_X = train_heart_df.iloc[:,:-1]
test_anime_X = test_heart_df.iloc[:,:-1]

#Label is the last column
train_heart_Y = train_heart_df.iloc[:,-1]
test_heart_Y = test_heart_df.iloc[:,-1]

# Train Decission Tree
tree_para = {'max_depth':(np.arange(10)+1).tolist(), 'min_samples_leaf':[1,2,3,4], 'n_estimators':[50,100,150,200], 'random_state':[0]}
dectree_heart = GridSearchCV(GradientBoostingClassifier(), tree_para, cv=5)
dectree_heart.fit(train_heart_X, train_heart_Y)

score = dectree_heart.score(test_anime_X, test_heart_Y)
print("Original GridSearch Heart Test Score: " + str(score))
print("Original Gridsearch Heart Params: " + str(dectree_heart.best_params_))
