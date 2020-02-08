import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve

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

title = "Decision Tree: Learning Curve for Anime"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=4)
image_loc = os.path.join(image_folder, 'anime_learning_curve.png')
plot_learning_curve(estimator, title, train_anime_X, train_anime_Y, ylim=(0.3, 0.6), cv=cv, n_jobs=4, image_loc=image_loc)

# Process Heart
print("Processing Heart...")
train_heart_loc = os.path.join(folder, "train_"+data_sets[1])
train_heart_df = pd.read_csv(train_heart_loc)
train_heart_X = train_heart_df.iloc[:,:-1]
train_heart_Y = train_heart_df.iloc[:,-1]

title = "Decision Tree: Learning Curve for Heart"
cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
estimator = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=4)
image_loc = os.path.join(image_folder, 'heart_learning_curve.png')
plot_learning_curve(estimator, title, train_heart_X, train_heart_Y, ylim=(0.45, 1.01), cv=cv, n_jobs=4, image_loc=image_loc)