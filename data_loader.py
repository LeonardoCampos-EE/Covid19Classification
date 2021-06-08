import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, train_dataframe, validation_dataframe, image_dir):

        self.train_dataframe = train_dataframe
        self.validation_dataframe = validation_dataframe
        self.image_dir = image_dir

        return

    def CreateImageGenerator(self):

        #Normalize the image and perform angle rotation as a data augmentation technique
        self.train_image_generator= ImageDataGenerator(
            rescale = 1.0/255.0,
            rotation_range = 30,
            fill_mode = 'nearest'
        )

        self.validation_image_generator = ImageDataGenerator(
            rescale = 1.0/255.0
        )

        return

    def CreateDataLoader(self):
        
        self.train_data_loader = self.train_image_generator.flow_from_dataframe(
            dataframe = self.train_dataframe,
            directory = self.image_dir,
            x_col = "filename",  # input images
            y_col = "finding", # labels
            class_mode = "binary", 
            batch_size = 16, # images per batch
            shuffle = False, 
            target_size = (416, 416) # width and height of output image
        )

        self.validation_data_loader = self.validation_image_generator.flow_from_dataframe(
            dataframe = self.validation_dataframe,
            directory = self.image_dir,
            x_col = "filename",  # input images
            y_col = "finding", # labels
            class_mode = "binary", 
            batch_size = 4, # images per batch
            shuffle = True, 
            target_size = (416, 416) # width and height of output image
        )

        return


if __name__ == '__main__':
    pass

