'''
This script is used to split the dataset into training, validation, and test sets.

Note: Splitting don't affect the images and labels in the original directory. 
      It only created txt files contiaining the name of the images and labels in the respective sets for training, validation, testing sets.

'''

import os 
import random

class DataSplit:
    def __init__(self, images_directory):
        """
        Initialize the DataSplit class.

        Args:
        - images_directory: The directory path where the images are stored.
        """
        self.images_directory = images_directory
        self.output_directory = None

    def make_all_txt(self, output_directory):
        """
        Create the 'all.txt' file containing the names of all the images. 
        'all.txt' is the text file that will be used to split the dataset into train, validation, and test sets.

        Args:
        - output_directory: The directory path where the 'all.txt' file will be saved.
        """
        image_names = [os.path.splitext(filename)[0] for filename in os.listdir(self.images_directory)]
        self.output_directory = output_directory       # output_directory of the txt files.
        self.all_txt_filepath = os.path.join(output_directory, 'all.txt')   # file path for the all.txt file.

        with open(self.all_txt_filepath, 'w') as file:
            file.write('\n'.join(image_names))

    def split_dataset(self, output_directory=None, train_percent=80, validation_percent=10, test_percent=None, shuffle=False):
        """
        Split the dataset into train, validation, and test sets.

        Args:
        - output_directory: The directory path where the train.txt, validation.txt, and test.txt files will be saved.
                            If not provided, the output_directory specified in make_all_txt() method will be used.
                            And if the make_all_txt() method is not called before running split_dataset, it will raise a ValueError.
        - train_percent: The percentage of images to be used for the train set.
        - validation_percent: The percentage of images to be used for the validation set.
        - test_percent: The percentage of images to be used for the test set. If not provided, it will be calculated based on the remaining percentage.
        - shuffle: Whether to shuffle the image names before splitting.

        Note:
        - The sum of train_percent, validation_percent, and test_percent should be 100.
        - If test_percent is not provided and train_percent + validation_percent is less than 100, the remaining percentage will be used for the test set. 
        - If test_percent is not provided, the related test.txt file won't be created.
        """
        # (1) Validate the output directory:
        if output_directory is None:
            if self.output_directory is None:
                raise ValueError('output_directory is not provided')
            elif self.output_directory: 
                output_directory = self.output_directory
        
        print(f"Train.txt, Validation.txt, and Test.txt files will be saved in {output_directory}")
        
        # (2) Validate the splitting percentages.
        try:
            # Check if the train_percent and validation_percent are integers or floats
            if not isinstance(train_percent, (int, float)):
                raise ValueError("train_percent should be an integer or a float number between 0 and 100")
            if not isinstance(validation_percent, (int, float)):
                raise ValueError("validation_percent should be an integer or a float number between 0 and 100")
            
            # Check if the train_percent is within the valid range of 0 to 100
            if not (0 <= train_percent <= 100):
                raise ValueError("train_percent should be between 0 and 100.")

            # Check if the validation_percent is within the valid range of 0 to 100
            if not (0 <= validation_percent <= 100):
                raise ValueError("validation_percent should be between 0 and 100.")

            # Check if the test_percent is either None or within the valid range of 0 to 100
            if test_percent is not None and not (0 <= test_percent <= 100):
                raise ValueError("test_percent should be between 0 and 100 or None.")

            # Check if the sum of train_percent and validation_percent is 100 or less
            if train_percent + validation_percent > 100:
                raise ValueError("The sum of train_percent and validation_percent should be 100 or less.")
            
            # Calculate the number for the test_percent if it is provided as None
            if test_percent is None:
                if train_percent + validation_percent < 100:
                    test_percent = 100 - (train_percent + validation_percent)
                elif train_percent + validation_percent == 100:
                    test_percent = 0

            # Check if the sum of train_percent, validation_percent, and test_percent is 100 or less
            if train_percent + validation_percent + (test_percent or 0) > 100:
                raise ValueError("The sum of train_percent, validation_percent, and test_percent should be 100 or less.")

            # If all conditions are met, print a success message
            print("The percentages are valid for splitting the dataset." +  "\nPercentages are:"
                f"\n    train_percent: {train_percent}", \
                f"\n    validation_percent: {validation_percent}", \
                f"\n    test_percent: {test_percent}")
            
        except ValueError as e:
            # If any of the conditions are not met, raise a ValueError with an error message
            print(f"Error: {str(e)}")

        # (3) Read image names from all.txt
        if self.all_txt_filepath is None:
            raise ValueError("all.txt file is not created yet. It can be created using object.make_all_txt() method, \n" \
                             "where object is an instance of DataSplit class and it the 'make_all_txt() method gets a single argument, \n" \
                             "which is the output directory for the all.txt file."
                             "e.g. object.make_all_txt('/path/to/images/directory')")

        with open(self.all_txt_filepath, 'r') as file:
            image_names = file.read().splitlines()

        # (4) Shuffle image names if shuffle is True
        if shuffle:
            random.shuffle(image_names)

        # (5) Calculate number of images for each set
        total_images = len(image_names)
        train_count = int(total_images * train_percent/100)
        print(f"Total images: {total_images}, Train images: {train_count}")
        validation_count = int(total_images * validation_percent/100)
        print(f"Validation images: {validation_count}")
        if test_percent != 0:  # if test_percent is not provided and train_percent + validation_percent is less than 100.
            test_count = total_images - train_count - validation_count


        # (6) Split image names into sets (train, validation, and test)
        train_names = image_names[:train_count]
        validation_names = image_names[train_count:train_count+validation_count]
        

        # (7) Save image names into train.txt, validation.txt, and test.txt
        # train.txt
        train_file = os.path.join(output_directory, 'train.txt')
        with open(train_file, 'w') as file:
            file.write('\n'.join(train_names))

        # validation.txt
        validation_file = os.path.join(output_directory, 'validation.txt')
        with open(validation_file, 'w') as file:
            file.write('\n'.join(validation_names))

        # test.txt
        if test_percent > 0:      # make the test.txt file only if there are images for the test set.
            test_names = image_names[train_count+validation_count:]
            test_file = os.path.join(output_directory, 'test.txt')
            with open(test_file, 'w') as file:
                file.write('\n'.join(test_names))



def test():
    """
    Test function to create the 'all.txt' file and split the dataset into train, validation, and test sets.
    """
    # Create an instance of DataSplit
    images_directory = "D:/Repos/UNet_semantic_segmentation/data/Aerial_Semantic_Segmenation_Drone_Dataset/dataset/semantic_drone_dataset/augmented_dataset/labels"
    data_splitter = DataSplit(images_directory)

    # Make the all.txt file
    output_directory = "D:/Repos/UNet_semantic_segmentation/data/Aerial_Semantic_Segmenation_Drone_Dataset/dataset/semantic_drone_dataset/augmented_dataset"
    data_splitter.make_all_txt(output_directory)

    # Split the dataset
    data_splitter.split_dataset(output_directory, train_percent=80, validation_percent=10, test_percent=10, shuffle=True)




if __name__ == "__main__":
    test() # This will run the test function when the script is run, where it will create the all.txt file and split the dataset into train, validation, and test sets with 80%, 10%, and 10% of the images, respectively.

        
        
        

