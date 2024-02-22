"""
This script is used to augment the data and save the augmented data into a new directory.
"""

import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt




class DataAugmentation:
    def __init__(self, input_dir):
        """
        Initialize the DataAugmentation class with input directory.

        Args:
            input_dir (str): Path to the input directory containing the original images.
        """
        self.input_dir = input_dir

    def _make_augmented_images(self, image):
        """
        Generate augmented images based on the given image.

        Args:
            image (numpy.ndarray): Original image to be augmented.

        Returns:
            list: List of augmented images.
        """
        augmented_images = []
        augmented_images.append(cv2.flip(image, 1))  # Horizontal Flip
        augmented_images.append(cv2.flip(image, 0))  # Vertical Flip
        augmented_images.append(cv2.flip(image, -1))  # Mirror
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))  # Rotate 90 degrees
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))  # Rotate 180 degrees
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))  # Rotate 270 degrees

        return augmented_images

    def augment_data(self, output_dir):
        """
        Apply data augmentation techniques to the images in the input directory and save the augmented images in the output directory.

        Args:
            output_dir (str): Path to the output directory to save the augmented images.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the list of image filenames in the input directory
        image_filenames = os.listdir(self.input_dir)

        # Create a progress bar
        progress_bar = tqdm(total=len(image_filenames), desc="Augmenting Images")

        # Iterate over images in the input directory
        for filename in image_filenames:
            # Load image
            image_path = os.path.join(self.input_dir, filename)
            image = cv2.imread(image_path)

            # Generate augmented images
            augmented_images = self._make_augmented_images(image)

            # Save augmented images
            for i, augmented_image in enumerate(augmented_images):
                output_filename = f"augmented_{i+1}_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, augmented_image)

            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

    def test_augmentation(self, index):
        """
        Test the data augmentation by displaying the original and augmented images.

        Args:
            index (int): Index of the image path to display and augment.
        """
        # Get the image path at the specified index
        image_path = os.listdir(self.input_dir)[index]
        original_image_path = os.path.join(self.input_dir, image_path)

        # Load original image
        original_image = cv2.imread(original_image_path)

        # Display original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        # Generate augmented images
        augmented_images = self._make_augmented_images(original_image)

        # Display augmented images
        for i, augmented_image in enumerate(augmented_images):
            plt.subplot(2, 3, i+2)
            plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Augmented Image {i+1}")

        plt.show()

if __name__ == "__main__":
    input_dir = "D:/Repos/UNet_semantic_segmentation/data/Aerial_Semantic_Segmenation_Drone_Dataset/dataset/semantic_drone_dataset/images"

    # Create an instance of DataAugmentation
    data_augmentation = DataAugmentation(input_dir)

    # Apply data augmentation and save augmented images
    output_dir = "D:/Repos/UNet_semantic_segmentation/data/Aerial_Semantic_Segmenation_Drone_Dataset/dataset/semantic_drone_dataset/augmented_dataset/images"
    data_augmentation.augment_data(output_dir)

    # Test the data augmentation by displaying the original and augmented images
    data_augmentation.test_augmentation()