
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

title = "K Nearest Neighbors: Learning Curve for Anime"
cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
estimator = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=22,weights='distance')
image_loc = os.path.join(image_folder, 'KNN_learning_curve_anime.png')
plot_learning_curve(estimator, title, train_anime_X, train_anime_Y, cv=cv, n_jobs=4, image_loc=image_loc)

# Use Gridsearch to analyse the hidden_layer_sizes of Neural
model_X = []
model_Y = []
model_STD = []
for i in range(5,30):
   tree_para = {'n_neighbors':[i],'weights':['distance'], 'algorithm':['ball_tree']}
   dectree = GridSearchCV(KNeighborsClassifier(), tree_para, cv=3)
   dectree.fit(train_anime_X, train_anime_Y)
   model_X.append(i)
   model_Y.append(dectree.best_score_)
   model_STD.append(np.average(dectree.cv_results_['std_test_score']))

best_idx = np.argmax(model_Y) + 5
image_loc = os.path.join(image_folder, 'KNN_N_model_complexity_anime.png')
plot_data(X=model_X, Y=model_Y, STD=model_STD, image_loc=image_loc, title='K Nearest Neighbors: n_neighbors Score for Anime', x_label='n_neighbors', x_best=best_idx)

# Process Heart
print("Processing Heart...")
train_heart_loc = os.path.join(folder, "train_"+data_sets[1])
train_heart_df = pd.read_csv(train_heart_loc)
train_heart_X = train_heart_df.iloc[:,:-1]
train_heart_Y = train_heart_df.iloc[:,-1]

title = "K Nearest Neighbors: Learning Curve for Heart"
cv = ShuffleSplit(n_splits=20, test_size=0.1, random_state=0)
estimator = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=11,weights='uniform')
image_loc = os.path.join(image_folder, 'KNN_learning_curve_heart.png')
plot_learning_curve(estimator, title, train_heart_X, train_heart_Y, cv=cv, n_jobs=4, image_loc=image_loc)

# Use Gridsearch to analyse the hidden_layer_sizes of Neural
model_X = []
model_Y = []
model_STD = []
for i in range(5,30):
   tree_para = {'n_neighbors':[i],'weights':['uniform'], 'algorithm':['ball_tree']}
   dectree = GridSearchCV(KNeighborsClassifier(), tree_para, cv=3)
   dectree.fit(train_heart_X, train_heart_Y)
   model_X.append(i)
   model_Y.append(dectree.best_score_)
   model_STD.append(np.average(dectree.cv_results_['std_test_score']))

best_idx = np.argmax(model_Y) + 5
image_loc = os.path.join(image_folder, 'KNN_model_complexity_heart.png')
plot_data(X=model_X, Y=model_Y, STD=model_STD, image_loc=image_loc, title='K Nearest Neighbors: n_neighbors Score for Heart', x_label='n_neighbors', x_best=best_idx)