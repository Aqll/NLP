#### Note: This is a copy of my own work initially developed uploaded in a private repo.

# How to install any libraries that are needed and for which you have the approval of the instructor.

1. Install python NLTK library using the following command in the terminal/bash shell:
>>>python3 -m pip install nltk

2. Install python nltk stopwords library:
>>>python3
>>>import nltk
>>>nltk.download('stopwords')

# Instructions on how to the your programs, including at least one example showing all parameters.

1. For Task 1: Running of the nbc training program. 

The first argument is the nbc_train program, the next arguments: the path to the json file with training data, the path to where the model tsv file created will be stored.
>>>python3 ./nbc/nbc_train.py ./data/train.json ./full_bbc_model.tsv   

2. For Task 2: Runing of the nbc inference program.

The first argument is the nbc_inference program, the next arguments: the path to the model created by my previous program, the path to a JSON test file like the one provided.
>>>python3 ./nbc/nbc_inference.py ./full_bbc_model.tsv ./data/test.json  

3. For Task 3: Running of the feature selection program.

The first argument is the path to the feature_selection program, the next arguments: the path to the training data, a value k, the path to where the filtered training file will be written.
>>>python3 ./nbc/feature_selection.py ./data/train.json 10 ./data/train_top_10.json

4. For Task 5: Running of the knn create model program.

The first argument is the path to the knn create model program, the next arguments: path to a JSON file with training data, path to a file where the document vectors in the training data will be written
>>>python3 ./knn/knn_create_model.py ./data/train.json ./bbc_doc_vectors.tsv

5. For Task 6: Running of the knn inference program.

The first argument is the path to the knn inference program, the next arguments: path to a TSV file with vectors computed as in the previous step, a value k, a path to a JSON test file like the one provided.
>>>python3 ./knn/knn_inference.py ./bbc_doc_vectors.tsv 11 ./data/test.json

# A discussion of which errors your programs detect and how these errors are handled.

1. For all the tasks, my program detects and print error messages if the program is called with a wrong number of arguments.
2. For all the tasks where a file is written as output, my program asks the user for confirmation before overwriting the file if the file already existed.
3. For all the tasks requiring a file as input, my program detects and print error messages if the program is called with an invalid path to a file. 


# Details on data structures, algorithms, and optimisation: 

Throughout my programs for all tasks, I have written comments on the functionality of the code, and meaning of the variables. The design of my code is easier to read, maintain, and debug. I've used dictionaries, including nested dictionaries, in all my programs to store/read information since it is much more efficient rather than looping through a list of lists.

# Resources consulted:

https://stackoverflow.com/questions/40496518/how-to-get-the-3-items-with-the-highest-value-from-dictionary
https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
