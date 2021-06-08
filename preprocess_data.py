import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
import random

class PreprocessData:

    def __init__(self, data_path):

        self.data_path = data_path

        # Object variable
        self.data_frame = self.CreateDataFrame()

        return

    def CreateDataFrame(self):

        data_frame = pd.read_csv(
            self.data_path,
            sep = ',',
            header = 0
        )

        return data_frame
    
    def FilterData(self):

        # Filter invalid images
        filter_images = self.data_frame["filename"].str.contains(r'.gz')

        # Interesting columns to keep
        keep_columns = ['patientid', 'finding', 'modality', 'filename']

        # Remove invalid images
        self.data_frame = self.data_frame[~filter_images]

        # Remove uninteresting columns
        self.data_frame = self.data_frame.filter(keep_columns)

        return

    def ConvertClasses2Bool(self):

        filter_classes = self.data_frame["finding"].str.contains("COVID")

        self.data_frame["finding"][filter_classes] = "1"
        self.data_frame["finding"][~filter_classes] = "0"

        return

    def PlotClassFrequency(self):

        filter_positive = self.data_frame['finding'] == True
        num_positive_cases = self.data_frame['finding'][filter_positive].value_counts()[1]
        num_negative_cases = self.data_frame['finding'][~filter_positive].value_counts()[0]

        frequency_positive_cases = 100*num_positive_cases/(num_positive_cases + num_negative_cases)
        frequency_negative_cases = 100*num_negative_cases/(num_positive_cases + num_negative_cases)

        plt.bar(
            x = ["Positive Cases", "Negative Cases"],
            height = [frequency_positive_cases, frequency_negative_cases],
            width = 0.8,
            color = "#113759",
            edgecolor = "#CD4446"
        )
        plt.show()

        return

    def SplitDatasetByPatients(self):

        unique_patients_list = self.data_frame['patientid'].unique()

        random.shuffle(unique_patients_list)

        self.data_frame['patientid'] = self.data_frame['patientid'].astype(np.int64)

        split_percentage = 0.95
        train_patients = unique_patients_list[:int(split_percentage*len(unique_patients_list))]
        validation_patients = unique_patients_list[int(split_percentage*len(unique_patients_list)):]

        self.train_dataset = self.data_frame[self.data_frame["patientid"].isin(train_patients)]
        self.validation_dataset = self.data_frame[self.data_frame["patientid"].isin(validation_patients)]

        return


if __name__ == '__main__':

    data_path = 'metadata.csv'
    object = PreprocessData(data_path)











