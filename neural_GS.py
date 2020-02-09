

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
import numpy as np

print("Begining Neural Networks GS process.")

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
#tree_para = {'activation':['logistic'], 'alpha':[0.1, 0.01, 0.001, 0.0001], 'beta_1':[0.5,0.6,0.7,0.8,0.9], 'random_state':[0], 'max_iter': [500]}
#dectree_anime = GridSearchCV(MLPClassifier(), tree_para, cv=5)
#dectree_anime.fit(train_anime_X, train_anime_Y)
#
#score = dectree_anime.score(test_anime_X, test_anime_Y)
#print("Original GridSearch Anime Test Score: " + str(score))
#print("Original Gridsearch Anime Params: " + str(dectree_anime.best_params_))
##Original Gridsearch Anime Params: {'activation': 'logistic', 'alpha': 0.0001, 'beta_1': 0.7, 'learning_rate': 'invscaling', 'max_iter': 500}

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
tree_para = {'activation':['logistic'], 'alpha':[0.1], 'beta_1':[0.7], 'random_state':[0], 'max_iter': [500]}
#dectree_heart = GridSearchCV(MLPClassifier(), tree_para, cv=5)
dectree_heart = MLPClassifier(activation='logistic', alpha=0.01, beta_1=0.7, max_iter=500, random_state=0)
dectree_heart.fit(train_heart_X, train_heart_Y)

score = dectree_heart.score(test_anime_X, test_heart_Y)
print("Original GridSearch Heart Test Score: " + str(score))
#print("Original Gridsearch Heart Params: " + str(dectree_heart.best_params_))
#Original Gridsearch Heart Params: {'activation': 'logistic', 'alpha': 0.01, 'beta_1': 0.7, 'max_iter': 500, 'random_state': 0}