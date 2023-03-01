import os
import torch
import torchvision.transforms as tf
from PIL import Image

def main():
    # Create new batches of images by applying random transformations to the original images
    previous_dir = os.listdir("Train_Images")
    output_dir = "Train_Images_Full"
    
    for card in previous_dir:
        for i in range(6):
            img = Image.open("Train_Images/" + card)
            img.save(output_dir + "/" + card.split(".")[0] + ".png")
            img = tf.RandomResizedCrop(size=(167, 258), scale=(0.9, 0.9))(img)
            img = tf.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img)
            img = tf.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(img)
            img = tf.RandomAdjustSharpness(sharpness_factor=0.1)(img)
            brightness_factor = torch.empty(1).uniform_(0.6, 1.4)
            img = tf.functional.adjust_brightness(img, brightness_factor)
            img.save(output_dir + "/" + card.split(".")[0] + "_" + str(i) + ".png")

if __name__ == "__main__":
    main()
    