#Name: Christopher Teeluckisngh
#ID: 89496
#AIML3001- Assignment2 - Speech Recognizer

#import Required libraries
import os
import argparse
import warnings
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc



#Read in audo files 
dir_path=r"C:\Users\Admin\Desktop\compEng\Year 3\Semester2\AIML\hmm-speech-recognition-0.1\hmm-speech-recognition-0.1\audio"
dir_subfolders=os.listdir(dir_path)

#get number of words being tested
num_of_words=len(dir_subfolders)
#get num of audio files in each word sub folder
num_of_files=len(os.listdir(os.path.join(dir_path, dir_subfolders[0])))

#initializing directory and file paths
file_names=np.zeros([num_of_files,num_of_words],dtype=object)
#get file names
for x in range(num_of_words):    
    file_names[:,x]=os.listdir(os.path.join(dir_path, dir_subfolders[x]))
    

#initialize model class
    
# Define a class to train the HMM
class ModelHMM(object):
    def __init__(self, num_components=4, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter
        #Define the covariance type and the type of HMM:
        self.cov_type = 'diag'
        self.model_name = 'GaussianHMM'
        #Initialize the variable in which we will store the models for each word:
        self.models = []
        #Define the model using the specified parameters:
        self.model = hmm.GaussianHMM(n_components=self.n_components,covariance_type=self.cov_type, n_iter=self.n_iter)
        
