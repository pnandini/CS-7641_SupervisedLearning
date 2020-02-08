import pandas as pd
import os

print("Begining Prep process.")

# This file moves the files along their required processes
folders = ["data_prep", "data_split", "data_final"]
data_sets = ["anime.csv", "heart.csv"]

# Anime file needs to be prepped and then split
# Original Location https://www.kaggle.com/CooperUnion/anime-recommendations-database
print("Processing Anime...")
prep_anime_loc = os.path.join(folders[0], data_sets[0])
prep_anime_df = pd.read_csv(prep_anime_loc)

# First build unique list of tags based on the data provided.
tags = set()
for index, row in prep_anime_df.iterrows():
   #If any of the columns is invalid then we will ignore it
   if row.isna().any():
      continue
   
   #This will all add all the unique tags to our set list
   row_tags = row['genre'].split(',')
   row_tags = [i.strip() for i in row_tags]
   tags |= set(row_tags)

# Convert to a list and sort so we can have the columns in alphabetical order
tags = list(tags)
tags.sort()

# Anime rating rules >= 90 => Great, else >= 80 Good, else >= 70 Mediocre, else >= 60 Bad, else Trash
def getRatingLabel(rating):
   rules = [9.,8.,7.,6.]

   for i in range(len(rules)):
      if rating >= rules[i]:
         return i

   # If nothing then one past the last rule will be the Trash label
   return len(rules)

processed_anime = {}

# Push every tag to the pre dataframe object
for tag in tags:
   processed_anime[tag] = []

# Push the last column which will be our custom rating function.
processed_anime['Rating'] = []

#Lets finally build our anime dataframe
for index, row in prep_anime_df.iterrows():
   #If any of the columns is invalid then we will ignore it
   if row.isna().any():
      continue

   #This will all add all the unique tags to our set list
   row_tags = row['genre'].split(',')
   row_tags = [i.strip() for i in row_tags]

   # If the anime has the tag then column will be set to 1 otherwise 0
   for tag in tags:
      if tag in row_tags:
         processed_anime[tag].append(1)
      else:
         processed_anime[tag].append(0)

   # Set the rating column to our special rating function
   processed_anime['Rating'].append(getRatingLabel(row['rating']))


processed_anime_df = pd.DataFrame(processed_anime, columns=processed_anime.keys())

# Save the anime processed file
split_anime_loc = os.path.join(folders[1], data_sets[0])
processed_anime_df.to_csv(split_anime_loc, encoding='utf-8', index=False)

# Heart File only needs to be split
# Original Location https://www.kaggle.com/ronitf/heart-disease-uci
print("Processing Heart...")
prep_heart_loc = os.path.join(folders[0], data_sets[1])
prep_heart_df = pd.read_csv(prep_heart_loc)
split_heart_loc = os.path.join(folders[1], data_sets[1])
prep_heart_df.to_csv(split_heart_loc, encoding='utf-8', index=False)

print("Done processing data!")