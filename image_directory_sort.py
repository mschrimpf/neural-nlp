import pandas as pd
import os.path
import shutil

meta = pd.read_pickle('meta.pkl')

dir_dict = {}

# generate dictionary for filename: desired subdirectory
for i in range(len(meta)):
    row = meta.loc[i,:]
    dir_dict[row['id']+'.png'] = [row['category'], row['obj'], row['var']]

folder_path = "images"
new_folder = "sorted_images"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# make copies of each image in new folders
for image in images:
    folder_list = dir_dict[image]
    folder_name = folder_list[0]+'/'+folder_list[1]+'/'+folder_list[2]

    new_path = os.path.join(new_folder, folder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old_image_path = os.path.join(folder_path, image)
    new_image_path = os.path.join(new_path, image)
    shutil.copy2(old_image_path, new_image_path)


