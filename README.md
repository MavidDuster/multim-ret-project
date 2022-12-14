# multim-ret-project - Group F
This project was implemented usning Python 3.9 using the following packages 
* numpy 
* pandas
* mathplotlib
* sklearn.metrics.pairwise for the similiarity compuation 
* sklearn.decomposition for PCA
* ast to deal with strings/lists from the tsv file
* tqdm


## Overview 
This repository contains the code and files for the course 344.038, KV Multimedia Search and Retrieval, taught by Markus Schedl, 2022W

We are tasked to create a framework for the retrieval of songs.
The retrieval system is suppoesd to use the lyrics as text features in addition to audio and image/video data.

## Data
We were provided with data from http://www.cp.jku.at/misc/MMSR/MMSR_WT22_Task2_Data.zip
The files in the zip are stored in a folder called data
We choose to focus on the features from Tf-idf and Bert, Blf_spectral,  Blf_correlation, Resnet and vgg19



## Preprocessing 
For sanity reasons we implemented a check if genre information is available, if it isn't entries from the dataset are removed
{run_check()}

## Data Analysis
Our very simple data analysis can be found in the data_analysis.py

## Retrieval
Our retrival method is found in model.py

## Eval
All evaluation method are found in evaluation.py

