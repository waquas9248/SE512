# Unmasking the Web of Deceit: An Analysis of Online Payment Fraud
## Data Mining 412/512 Team 4 Project Report

### Authors
* Shawn Eidem - shawn.eidem@go.stcloudstate.edu
* William Ortman - william.ortman@go.stcloudstate.edu
* Noah Blon - noah.blon@go.stcloudstate.edu
* Ashhad Waquad Syed - ashhadwaquas.syed@go.stcloudstate.edu

## Requirements:
* Python 3.11.4
  
## Setup

1. Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Downloading the Dataset
The dataset if available on Kaggle here: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset.

You can either download the dataset and place it in the data folder with the name: online-payments-fraud-detection-dataset.csv

Or you can setup Kaggle API key as described here: https://www.kaggle.com/docs/api.  If you use this method, the notebook will download the data automatically for you the first time you run it.



## Files
There are two files, one is the Fraud Detection Jupyter Notebook which contains our report. The second is the Algorithms.py file. This can be run against the unsampled dataset, but takes awhile as the SVM algorithm is slow given the size of the dataset. This file was used to generate metrics which we statically included in our report for SVM.
