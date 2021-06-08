import numpy as np
import cv2

from data_loader import DataLoader
from preprocess_data import PreprocessData

if __name__ == '__main__':

    data_path = 'metadata.csv'

    preprocessor = PreprocessData(data_path)

    preprocessor.FilterData()
    preprocessor.ConvertClasses2Bool()
    preprocessor.SplitDatasetByPatients()

    train_dataframe = preprocessor.train_dataset

    print(train_dataframe.head())

    validation_dataframe = preprocessor.validation_dataset

    data_loader = DataLoader(train_dataframe, validation_dataframe, "/home/leonardo/ApolloTraining/Covid19Classification/images")

    data_loader.CreateImageGenerator()
    data_loader.CreateDataLoader()

    train_dataloader = data_loader.train_data_loader
    valid_dataloader = data_loader.validation_data_loader


    image, label = train_dataloader.__getitem__(5)
    cv2.namedWindow("T", 0)
    cv2.imshow("T", image[0])
    cv2.waitKey(0)
    print(label)









