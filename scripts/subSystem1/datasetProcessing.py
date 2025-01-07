# Imports
import os
import pandas as pd
from feat import Detector

# Files and Directories
dataset = "../DiffusionFER/DiffusionEmotion_S"
input_images_original = os.path.join(dataset, "original")
input_images_cropped = os.path.join(dataset, "cropped")
input_data_sheet = os.path.join(dataset, "dataset_sheet.csv")

output = "processed"
output_aus_original = os.path.join(output, "facial_features_original.csv")
output_aus_cropped = os.path.join(output, "facial_features_cropped.csv")

# Load annotations from csv file
data_sheet = pd.read_csv(input_data_sheet)

# Initialize the feature detector
detector = Detector(device='cpu')

def main():

    aus_original = []
    aus_cropped = []

    # Iterate over the data sheet
    for index, row in data_sheet.head(50).iterrows():

        image_path_full = row['subDirectory_filePath']
        image_name = image_path_full.split('/')[-1]
        folder_name = image_path_full.split('/')[-2]
        image_path_original = os.path.join(input_images_original, folder_name, image_name)
        image_path_cropped = os.path.join(input_images_cropped, folder_name, image_name)

        # Detect faces and emotions
        try:
            predictions_original = detector.detect_image([image_path_original])
            if not predictions_original.aus.empty:
                new_row_original = pd.concat([row.to_frame().T.reset_index(drop=True), predictions_original.aus],
                                             axis=1)
                new_row_original['expression'] = folder_name
                aus_original.append(new_row_original)
        except Exception as e:
            print(e)

        try:
            predictions_cropped = detector.detect_image([image_path_cropped])
            if not predictions_cropped.aus.empty:
                new_row_cropped = pd.concat([row.to_frame().T.reset_index(drop=True), predictions_cropped.aus], axis=1)
                new_row_cropped['expression'] = folder_name
                aus_cropped.append(new_row_cropped)
        except Exception as e:
            print(e)

    # Combine all rows into single DataFrames and save them
    if aus_original:
        df_original = pd.concat(aus_original, ignore_index=True)
        df_original.dropna().to_csv(output_aus_original, index=False)

    if aus_cropped:
        df_cropped = pd.concat(aus_cropped, ignore_index=True)
        df_cropped.dropna().to_csv(output_aus_cropped, index=False)

if __name__ == '__main__':
    main()
