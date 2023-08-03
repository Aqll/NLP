
# Text Classification Project

## Overview

This project implements a Naive Bayes classifier for text classification tasks. The Naive Bayes classifier is a popular machine learning algorithm used in natural language processing and text classification. It makes the assumption of independence among features, which simplifies the classification process. The project provides functionalities to train the classifier on labeled text data and perform text classification on new data using the trained model.


<img align="left" width = 405 src="https://upload.wikimedia.org/wikipedia/commons/b/b4/Naive_Bayes_Classifier.gif"/>  

<img align="right" height=330 width = 405 src="https://importq.files.wordpress.com/2017/11/knn_mov5.gif?w=640&zoom=2"/>  


## Objective

The main objective of this project is to create a simple and efficient text classification tool based on the Naive Bayes algorithm. The specific objectives are as follows:

- Develop a robust Naive Bayes classifier that can handle both binary and multiclass classification tasks.
- Preprocess and tokenize text data to create feature vectors suitable for classification.
- Train the classifier on labeled text data to learn the underlying patterns in the data.
- Provide an easy-to-use interface for users to perform text classification on new data using the trained classifier.
- Evaluate the performance of the classifier using various metrics to assess its accuracy and effectiveness.

## Features

- Train a Naive Bayes classifier on labeled text data.
- Perform text classification on new data using the trained classifier.
- Evaluate the classifier's performance using various metrics.
- Tokenize and preprocess text data to create feature vectors.
- Handle both binary and multiclass classification tasks.

## Details on Data Structures, Algorithms, and Optimization

### Data Structures:
1. **Feature Vectors:** The text data is preprocessed and tokenized to create feature vectors. These feature vectors represent the words and their frequencies in the text. We use Python dictionaries to efficiently store these feature vectors, where each word corresponds to a key, and its frequency in the text is the value.

### Algorithms:
1. **Naive Bayes Classifier:** The main algorithm used in this project is the Naive Bayes Classifier, which is a probabilistic machine learning algorithm. It leverages Bayes' theorem and makes the naive assumption of independence among features given the class label. The classifier calculates the posterior probability of each class given the input text and selects the class with the highest probability as the predicted class.
2. **k-Nearest Neighbors (kNN):** The k-Nearest Neighbors algorithm is another essential algorithm employed in the text classifier project. It is a non-parametric and instance-based learning algorithm used for classification tasks. In kNN, the class of a data point is determined by the majority class among its k nearest neighbors in the feature space. The distance metric used (e.g., Euclidean distance) plays a crucial role in finding the nearest neighbors.



### Optimization:
1. **Multinomial Naive Bayes:** We use the Multinomial Naive Bayes variant, which is specifically suited for text classification tasks, as it works well with discrete features like word counts. This variant is optimized for handling text data and efficiently computing probabilities for multiple classes.

2. **Sparse Representation:** Since the text data typically contains a large number of unique words, most feature vectors are sparse, i.e., most entries are zero. To optimize memory usage and computational efficiency, we use sparse representations of feature vectors, which only store the non-zero elements.

3. **Log Probabilities:** To prevent underflow issues that can arise when multiplying probabilities, we work with log probabilities. Since probabilities are small numbers, they can lead to numerical instability when multiplied repeatedly. By converting probabilities to log space and performing additions instead of multiplications, we can avoid underflow and maintain numerical stability.

4. **Vectorization:** To speed up computations, we utilize vectorized operations from libraries like NumPy. Vectorized operations allow us to perform computations on entire arrays of data at once, leading to significant performance improvements compared to using traditional loops.

5. **Data Preprocessing:** Efficient data preprocessing techniques, such as tokenization, stop-word removal, and word stemming, are employed to clean the text data and reduce the feature space, leading to faster training and inference times.

These data structures, algorithms, and optimization techniques contribute to the overall efficiency and effectiveness of the Naive Bayes Text Classification project. By carefully choosing appropriate data structures and applying optimization strategies, the classifier can handle large text datasets and provide reliable predictions efficiently.

## Files

The repository contains the following files:

- `/nbc/nbc_train.py`: Script for training the Naive Bayes Classifier.
- `/nbc/nbc_inference.py`: Script for using the trained Naive Bayes Classifier for text classification.
- `nbc/feature_selection.py`: Script for performing feature selection in text data.
- `knn/knn_train.py`: Script for training the k-Nearest Neighbors Classifier.
- `knn/knn_inference.py`: Script for using the trained k-Nearest Neighbors Classifier for text classification.
- `data/`: Directory containing sample training and testing data.


## How to install any libraries that are needed.

1. Install python NLTK library using the following command in the terminal/bash shell:
>>>python3 -m pip install nltk

2. Install python nltk stopwords library:
>>>python3
>>>import nltk
>>>nltk.download('stopwords')

## Instructions on how to run, including example showing all parameters.

1. Running of the nbc training program. 

The first argument is the nbc_train program, the next arguments: the path to the json file with training data, the path to where the model tsv file created will be stored.
>>>python3 ./nbc/nbc_train.py ./data/train.json ./full_bbc_model.tsv   

2. Runing of the nbc inference program.

The first argument is the nbc_inference program, the next arguments: the path to the model created by my previous program, the path to a JSON test file like the one provided.
>>>python3 ./nbc/nbc_inference.py ./full_bbc_model.tsv ./data/test.json  

3. Running of the feature selection program.

The first argument is the path to the feature_selection program, the next arguments: the path to the training data, a value k, the path to where the filtered training file will be written.
>>>python3 ./nbc/feature_selection.py ./data/train.json 10 ./data/train_top_10.json

4. Running of the knn create model program.

The first argument is the path to the knn create model program, the next arguments: path to a JSON file with training data, path to a file where the document vectors in the training data will be written
>>>python3 ./knn/knn_create_model.py ./data/train.json ./bbc_doc_vectors.tsv

5. Running of the knn inference program.

The first argument is the path to the knn inference program, the next arguments: path to a TSV file with vectors computed as in the previous step, a value k, a path to a JSON test file like the one provided.
>>>python3 ./knn/knn_inference.py ./bbc_doc_vectors.tsv 11 ./data/test.json

## A discussion of which errors the program detects and how these errors are handled.

1. For all the tasks, the program detects and print error messages if the program is called with a wrong number of arguments.
2. For all the tasks where a file is written as output, the program asks the user for confirmation before overwriting the file if the file already existed.
3. For all the tasks requiring a file as input, the program detects and print error messages if the program is called with an invalid path to a file. 
