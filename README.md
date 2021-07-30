# Jump-QC

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


## Introduction to the Project 
Image QC is currently a tedious, mostly-manual process of generating training data on a per-experiment basis. E.g. for a 16-plate dataset approximately 1/3 of the hands-on time goes into this step. The goal of the project is to create a package that takes test image set, and give a quality score to each image and generate a pdf report with examples of LQ images


### Description of the Project
The project files are organised in four folders, namely, code/, data/, notebooks/ and results/. The description of files in each of these folders is given below:
1. code/
The code/ folder contains the training and evaluation files of the baseline deep learning models and transformer models. For transformer models we have three files, 'train_qc_transformer.py', 'eval_qc_transformer.py', and 'test_qc_transformer.py'. For baseline deep learning models we have three files, 'train_qc_baseline.py', 'eval_qc_baseline.py', and 'test_qc_baseline.py'. For running the files, all the user needs to do is specify the necessary paths of the saved model and the output.
2. data/
The data/ folder consisted of the annotations from Broad and the seven partners. It consisted of the following files:
   a. qc_annotations.parquet
	This file consists of all the annotations from Broad and the partners. This is the main file to be used for data analysis of this project.
   b. train_validation.csv
	This file consists of the required information for the files present in the training and validation dataset. This is the subset of the image files found in the qc_annotations.parquet file.
   c. broad_annotations/
	This folder consists of the all the files obtained for the annotation of images belonging to Broad.
   d. partner_annotations/
	This folder consists of the all the files obtained for the annotation of images belonging to all the partners.
3. notebooks/
	This folder consists of three folders, namely, data_processing, data_creation and inspect-results. Each of the sub-folder consists the required notebooks that were used for the task and the description of each notebook can be seen in the notebook itself.
4. results/
The results/ folder was where the outputs were stored. It was divided in three folders, namely, data/, model/ and test_dataset/
   a. results/data
       This folder contained the distribution of data from the entire dataset and also among the training and validation dataset. We obtained data among four classes (good, blurry, empty and debris) and from seven partners (Bayer, Merck, MPI, Servier, Pfizer, Ksilink, AstraZeneca and Broad Institute). The files corresponding to distribution of entire data were: data_good.png, data_blurry.png, data_empty.png and data_debris.png. They represent the number of images each partner contributed for each class. Similarly, train_good.png, train_blurry.png, train_empty.png and train_debris.png were files corresponding to distribution of training data while val_good.png, val_blurry.png, val_empty.png and val_debris.png were files corresponding to distribution of validation data.
    b. results/model
       This folder contained the model files and the corresponding validation loss vs epochs and validation accuracy vs epochs curves for the three models used in this task. It contained three folders, namely, baseline_cpa/, baseline_dl/ and transformer/ which corresponds to the three models used. The folders contain the best model files which were obtained after the training process and the corresponding validation curves.
    c. results/test_dataset
       This folder contains the three model predictions on the test dataset. The test dataset used was Stain5_CondC from Broad Institute which consisted of 86385 images. The folder consists of three files, namely, test_baseline_cpa.csv, test_baseline.csv and test_transformer.csv which have the corresponding model predictions for each file in the test dataset.
    d. results/misc
       This folder consists of the files used for the work during the project. These are not relevant and thus are put in the misc/ folder.


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.
