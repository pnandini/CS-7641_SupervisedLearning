import pandas as pd
import os

print("Begining Split process.")

# This file moves the files along their required processes
folders = ["data_prep", "data_split", "data_final"]
data_sets = ["anime.csv", "heart.csv"]

# Process Anime
print("Processing Anime...")
split_anime_loc = split_anime_loc = os.path.join(folders[1], data_sets[0])
split_anime_df = pd.read_csv(split_anime_loc)

# Shuffle the data
split_anime_df = split_anime_df.sample(frac=1, random_state=0).reset_index(drop=True)

# Split the data
anime_train_size = len(split_anime_df) - int(len(split_anime_df) * 0.1)
final_test_anime_loc  = os.path.join(folders[2], "test_"+data_sets[0])
final_train_anime_loc = os.path.join(folders[2], "train_"+data_sets[0])

# Save
split_anime_df[anime_train_size:].to_csv(final_test_anime_loc,header=False)
split_anime_df[:anime_train_size].to_csv(final_train_anime_loc,header=False)


# Process Heart
print("Processing Heart...")
split_heart_loc = split_anime_loc = os.path.join(folders[1], data_sets[1])
split_heart_df = pd.read_csv(split_heart_loc)

# Shuffle the data
split_heart_df = split_heart_df.sample(frac=1, random_state=0).reset_index(drop=True)

# Split the data
heart_train_size = len(split_heart_df) - int(len(split_heart_df) * 0.1)
final_test_heart_loc  = os.path.join(folders[2], "test_"+data_sets[1])
final_train_heart_loc = os.path.join(folders[2], "train_"+data_sets[1])

# Save
split_heart_df[heart_train_size:].to_csv(final_test_heart_loc,header=False)
split_heart_df[:heart_train_size].to_csv(final_train_heart_loc,header=False)

print("Done splitting data!")