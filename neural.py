
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from plot_learning_curve import plot_learning_curve
from plot_data import plot_data

 # This file moves the files along their required processes
folder = "data_final"
image_folder = "images"
data_sets = ["anime.csv", "heart.csv"]

# Process Anime
print("Processing Anime...")
train_anime_loc = os.path.join(folder, "train_"+data_sets[0])
train_anime_df = pd.read_csv(train_anime_loc)
train_anime_X = train_anime_df.iloc[:,:-1]
train_anime_Y = train_anime_df.iloc[:,-1]

#title = "Neural Network: Learning Curve for Anime"
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = MLPClassifier(activation='logistic',alpha=0.0001,beta_2=0.7,max_iter=500,random_state=0)
#image_loc = os.path.join(image_folder, 'Neural_learning_curve_anime.png')
#plot_learning_curve(estimator, title, train_anime_X, train_anime_Y, ylim=(0.65, 0.75), cv=cv, n_jobs=4, image_loc=image_loc)

# Use Gridsearch to analyse the hidden_layer_sizes of Neural
model_X = []
model_Y = []
model_STD = []
for i in range(5, 16):
   param = i * 10
   tree_para = {'hidden_layer_sizes':[(param,)], 'activation':['logistic'], 'alpha':[0.0001], 'beta_1':[0.7], 'random_state':[0], 'max_iter': [500]}
   dectree = GridSearchCV(MLPClassifier(), tree_para, cv=5)
   dectree.fit(train_anime_X, train_anime_Y)
   model_X.append(param)
   model_Y.append(dectree.best_score_)
   model_STD.append(np.average(dectree.cv_results_['std_test_score']))

best_idx = ((np.argmax(model_Y) + 1) * 10) + 40
image_loc = os.path.join(image_folder, 'Neural_hidden_layer_model_complexity_anime.png')
plot_data(X=model_X, Y=model_Y, STD=model_STD, image_loc=image_loc, title='Neural Network: Hidden Layer Score for Anime', x_label='Hidden Layer', x_best=best_idx)

## Process Heart
#print("Processing Heart...")
#train_heart_loc = os.path.join(folder, "train_"+data_sets[1])
#train_heart_df = pd.read_csv(train_heart_loc)
#train_heart_X = train_heart_df.iloc[:,:-1]
#train_heart_Y = train_heart_df.iloc[:,-1]
#
#title = "Neural Network: Learning Curve for Heart"
#cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
#estimator = MLPClassifier(activation='logistic', alpha=0.01, beta_1=0.7, max_iter=1000, random_state=0)
#image_loc = os.path.join(image_folder, 'Neural_learning_curve_heart.png')
#plot_learning_curve(estimator, title, train_heart_X, train_heart_Y, ylim=(0.45, 1.01), cv=cv, n_jobs=4, image_loc=image_loc)
#
## Use Gridsearch to analyse the hidden_layer_sizes of Neural
#model_X = []
#model_Y = []
#model_STD = []
#for i in range(1, 10):
#   param = i * 10
#   tree_para = {'hidden_layer_sizes':[(param,)], 'activation':['logistic'], 'alpha':[0.0001], 'beta_1':[0.7], 'random_state':[0], 'max_iter': [1000]}
#   dectree = GridSearchCV(MLPClassifier(), tree_para, cv=5)
#   dectree.fit(train_heart_X, train_heart_Y)
#   model_X.append(param)
#   model_Y.append(dectree.best_score_)
#   model_STD.append(np.average(dectree.cv_results_['std_test_score']))
#
#best_idx = (np.argmax(model_Y) + 1) * 10
#image_loc = os.path.join(image_folder, 'Neural_hidden_layer_model_complexity_heart.png')
#plot_data(X=model_X, Y=model_Y, STD=model_STD, image_loc=image_loc, title='Neural Network: Hidden Layer Score for Heart', x_label='Hidden Layer', x_best=best_idx)