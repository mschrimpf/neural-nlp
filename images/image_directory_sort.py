import argparse
import pandas as pd
import os.path
import shutil

if __name__ == '__main__':
    dirpath = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default=os.path.join(dirpath, 'unsorted'),
                        help='Directory that the unsorted images are stored in')
    parser.add_argument('--target_dir', type=str, default=os.path.join(dirpath, 'sorted'),
                        help='Directory to store the sorted images in')
    args = parser.parse_args()
    print("Running with args", args)

    images_metadata = pd.read_pickle(os.path.join(dirpath, 'meta.pkl'))

    image_sourcepaths = [os.path.join(args.source_dir, f) for f in os.listdir(args.source_dir) if f.endswith('.png')]
    for image_sourcepath in image_sourcepaths:
        # image info
        image_basename = os.path.basename(image_sourcepath)
        image_metadata = images_metadata.loc[images_metadata.id == os.path.splitext(image_basename)[0]]
        assert len(image_metadata) == 1
        image_metadata = image_metadata.iloc[0]  # get the single cell wrapped in the DataFrame
        # target directory
        target_directory_name = os.path.join(image_metadata['category'], image_metadata['obj'], image_metadata['var'])
        target_directory = os.path.join(args.target_dir, target_directory_name)
        if not os.path.isdir(target_directory):
            os.makedirs(target_directory)
        # copy
        target_image_path = os.path.join(target_directory, os.path.basename(image_sourcepath))
        shutil.copy2(image_sourcepath, target_image_path)
