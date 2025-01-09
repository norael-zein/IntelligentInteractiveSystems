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
    """
    Processes the DiffusionFER dataset to be used for model training.
    Extracts action units (AUs) from the images using the py-feat Detector class.
    Two passes to process both the original and cropped images simultaneously.
    """

    # Store data as single-row dataframes
    aus_original = []
    aus_cropped = []

    # Iterate over the data sheet
    for index, row in data_sheet.iterrows():

        # Construct the file paths
        image_path_full = row['subDirectory_filePath']
        image_name = image_path_full.split('/')[-1]
        folder_name = image_path_full.split('/')[-2]
        image_path_original = os.path.join(input_images_original, folder_name, image_name)
        image_path_cropped = os.path.join(input_images_cropped, folder_name, image_name)

        # Detect faces and emotions for images in original folder
        try:
            predictions_original = detector.detect_image([image_path_original])

            # Check if a face was detected
            if not predictions_original.aus.empty:
                # Create new dataframe by appending the action units to the information from the corresponding row
                new_row_original = pd.concat([row.to_frame().T.reset_index(drop=True), predictions_original.aus],
                                             axis=1)
                # Replace numerical value with string label
                new_row_original['expression'] = folder_name
                # Append dataframe
                aus_original.append(new_row_original)
        except Exception as e:
            print("Error (DatasetProcessing): {}".format(e))

        # Detect faces and emotions for images in cropped folder
        try:
            predictions_cropped = detector.detect_image([image_path_cropped])

            # Check if a face was detected
            if not predictions_cropped.aus.empty:
                # Create new dataframe by appending the action units to the information from the corresponding row
                new_row_cropped = pd.concat([row.to_frame().T.reset_index(drop=True), predictions_cropped.aus], axis=1)
                # Replace numerical value with string label
                new_row_cropped['expression'] = folder_name
                # Append dataframe
                aus_cropped.append(new_row_cropped)
        except Exception as e:
            print("Error (DatasetProcessing): {}".format(e))

    """
    Combine all rows into single DataFrames and save them.
    Use .dropna() to filter out rows with invalid or null entries.
    Happens as result of inconsistent dataset, original folder has more images than cropped folder
    """
    if aus_original:
        df_original = pd.concat(aus_original, ignore_index=True)
        df_original.dropna().to_csv(output_aus_original, index=False)

    if aus_cropped:
        df_cropped = pd.concat(aus_cropped, ignore_index=True)
        df_cropped.dropna().to_csv(output_aus_cropped, index=False)

if __name__ == '__main__':
    main()
