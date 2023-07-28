import json	
import nltk
import string
import csv	
import sys
import math
import ast #to convert string representation of dictionary to actual dictionary
from nltk.corpus import stopwords
from collections import Counter

#Function for pre-processing the data
def tokenize_function(data):
	terms = {} #dictionary containing all terms found in a document
	category={} #list containing all categories
	doc_id = 0 #used to identify a document
	#I removed stopword because these term do not provide information about the class
	stop_words = stopwords.words('english')
	stopwords_dict = Counter(stop_words) #using dictionary to run faster than using list

	#Looping through all the data
	for each_doc in data:
		doc_id +=1
		#tokenization/pre-processing
		string1 = str(each_doc['text'])
		tokenizer = nltk.RegexpTokenizer(r"\w+")
		tokens = tokenizer.tokenize(string1) 
		tokens = [word for word in tokens if not word in stopwords_dict]

		#saving the document in a dictionary
		terms[doc_id,each_doc['category']] = tokens

		if each_doc['category'] not in category:
			category[each_doc['category']] = each_doc['category']

	terms['all_categories'] = category
	
	return terms 

#Function to calculate tp,fn,tn,fp statistis
def calculate_statistics(classes, predicted_class_each_doc):
	positives_negatives = {} #positives_negatives stores the value of TP/TN/FP/FN for each class

	#initialising all values to zero
	for each_class in classes:
		positives_negatives[each_class, 'true_positive'] = 0
		positives_negatives[each_class, 'true_negatives'] = 0
		positives_negatives[each_class, 'false_positives'] = 0
		positives_negatives[each_class, 'false_negatives'] = 0

	for each_doc in predicted_class_each_doc:
		#each_doc is the (doc_id,class)
		if each_doc[1] == 'entertainment':
			if predicted_class_each_doc[each_doc] == 'entertainment':
				positives_negatives['entertainment', 'true_positive'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'sport':
				positives_negatives['entertainment', 'false_negatives'] += 1
				positives_negatives['sport', 'false_positives'] += 1

				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'politics':
				positives_negatives['entertainment', 'false_negatives'] += 1
				positives_negatives['politics', 'false_positives'] += 1

				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'tech':
				positives_negatives['entertainment', 'false_negatives'] += 1
				positives_negatives['tech', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'business':
				positives_negatives['entertainment', 'false_negatives'] += 1
				positives_negatives['business', 'false_positives'] += 1

				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1

		elif each_doc[1] == 'sport':
			if predicted_class_each_doc[each_doc] == 'sport':
				positives_negatives['sport', 'true_positive'] += 1
				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'entertainment':
				positives_negatives['sport', 'false_negatives'] += 1
				positives_negatives['entertainment', 'false_positives'] += 1

				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'politics':
				positives_negatives['sport', 'false_negatives'] += 1
				positives_negatives['politics', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'tech':
				positives_negatives['sport', 'false_negatives'] += 1
				positives_negatives['tech', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'business':
				positives_negatives['sport', 'false_negatives'] += 1
				positives_negatives['business', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1

		elif each_doc[1] == 'politics':
			if predicted_class_each_doc[each_doc] == 'politics':
				positives_negatives['politics', 'true_positive'] += 1
				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'entertainment':
				positives_negatives['politics', 'false_negatives'] += 1
				positives_negatives['entertainment', 'false_positives'] += 1

				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'sport':
				positives_negatives['politics', 'false_negatives'] += 1
				positives_negatives['sport', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'tech':
				positives_negatives['politics', 'false_negatives'] += 1
				positives_negatives['tech', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'business':
				positives_negatives['politics', 'false_negatives'] += 1
				positives_negatives['business', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['tech', 'true_negatives'] += 1
				
		elif each_doc[1] == 'business':
			if predicted_class_each_doc[each_doc] == 'business':
				positives_negatives['business', 'true_positive'] += 1
				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['tech', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'entertainment':
				positives_negatives['business', 'false_negatives'] += 1
				positives_negatives['entertainment', 'false_positives'] += 1

				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['tech', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'sport':
				positives_negatives['business', 'false_negatives'] += 1
				positives_negatives['sport', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['tech', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'tech':
				positives_negatives['business', 'false_negatives'] += 1
				positives_negatives['tech', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1

			elif predicted_class_each_doc[each_doc] == 'politics':
				positives_negatives['business', 'false_negatives'] += 1
				positives_negatives['politics', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1

		elif each_doc[1] == 'tech':
			if predicted_class_each_doc[each_doc] == 'tech':
				positives_negatives['tech', 'true_positive'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'entertainment':
				positives_negatives['tech', 'false_negatives'] += 1
				positives_negatives['entertainment', 'false_positives'] += 1

				
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'sport':
				positives_negatives['tech', 'false_negatives'] += 1
				positives_negatives['sport', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				positives_negatives['business', 'true_negatives'] += 1

			elif predicted_class_each_doc[each_doc] == 'business':
				positives_negatives['tech', 'false_negatives'] += 1
				positives_negatives['business', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['politics', 'true_negatives'] +=1
				
			elif predicted_class_each_doc[each_doc] == 'politics':
				positives_negatives['tech', 'false_negatives'] += 1
				positives_negatives['politics', 'false_positives'] += 1

				positives_negatives['entertainment', 'true_negatives'] += 1
				positives_negatives['sport', 'true_negatives'] += 1
				positives_negatives['business', 'true_negatives'] += 1

	return positives_negatives

#Function for calculating precision, recall, F1, microaverage, macroaverage and printing the output to STDOUT
def print_statistics(positives_negatives, classes):

	#intialise sum of F1, TP, FP, FN of all classes and number of classes to zero
	sum_F1 = sum_TP = sum_FP = sum_FN = num_class = 0 
	for each_class in classes:
		num_class +=1
		TP = positives_negatives[each_class, 'true_positive']
		FP = positives_negatives[each_class, 'false_positives']
		FN = positives_negatives[each_class, 'false_negatives']
		TN = positives_negatives[each_class, 'true_negatives']

		if (TP + FP) != 0:
			precision = TP/(TP+FP)
		else:
			precision = 0
		recall = TP/(TP+FN)
	
		if (precision+recall) != 0:
			F1 = (2*precision*recall)/(precision+recall)
		else: 
			F1 = 0

		sum_TP += TP
		sum_FP += FP
		sum_FN += FN
		sum_F1 += F1
		print("Class: "+each_class+"\t TP: "+str(TP), end = " ") 
		print("TN: ", TN, end = " ")
		print("FP: ", FP, end = " ")
		print("FN: ", FN, end = " ")
		print("P: ", precision, end = " ")
		print("R: ", recall, end = " ")
		print("F1: ", F1)

	macroaverage = sum_F1/num_class
	precision_aggregate = sum_TP/(sum_TP+sum_FP)
	recall_aggregate = sum_TP/(sum_TP+sum_FN) 
	microaverage = (2*precision_aggregate*recall_aggregate)/(precision_aggregate+recall_aggregate)
	print("Micro-averaged F1: ", microaverage)
	print("Macro-averaged F1: ", macroaverage)

	return 0

#Function to read the training data and store in a dictionary
def read_training_data(path_vector_file):

	#key=(class, doc_id)
	#value = vector for the document
	knn_model = {} #dictionary containing vector for each doc
	doc_id = 0
	with open(path_vector_file, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = '\t')
		for row in reader:
			if row['idf/vector'] == 'vector':
				doc_id +=1
				knn_model[row['class(c)'], doc_id] = row['vector']

	return knn_model

#Function to get the idf values for each term in the testing data
def get_idf(terms, N):
	idf_terms = {} #dictionary to hold each distinct term found in the document and their respective idf
				   #key = term, value = idf of term
	#df of a term t is given by len(terms["t"])
	#terms is a dictionary with key: distinct terms, and value: (ids,classes) in which the term appears
	for each_term in terms:

		idf_before_log = N/len(terms[each_term]) 
		idf = math.log(idf_before_log, 10)
		idf_terms[each_term] = idf

	return idf_terms

#Function to get the vector for each documents in the testing data
def get_vector(classes, terms, idf_terms, docs):
	vector = {} #vector is a dictionary with key = (docs, class), value = (terms occuring in that doc, the weight)
	for each_doc in docs:
		weight = {} #to store final weight for every term in the document
		vector[each_doc] = {} #each_doc[0] is the doc id, each_doc[1] is the class of that doc
		#get all term frequency for all terms in the document
		#weight[each_term] is the term frequency of the each_term
		for each_term in docs[each_doc]:
			if each_term not in weight:
				weight[each_term] = 1 
			else:
				weight[each_term] += 1

		for each_term in weight:
			weight[each_term] =  idf_terms[each_term] * (1+math.log(weight[each_term], 10)) #(1 + log tf)

			if weight[each_term] != 0 :
				if each_term not in vector[each_doc]:
						vector[each_doc][each_term] =  weight[each_term]

	return vector

#Function for pre-processing the data
def tokenize_function(data):
	terms = {} #dictionary containing all terms in documents found in the corpus
	category={} #dictionary containing all categories
	doc_id = 0 #used to identify a document
	docs = {} #dictionary containing all documents and their tokens
	#Looping through all the data
	for each_doc in data:
		doc_id +=1
		#tokenization/pre-processing
		string1 = str(each_doc['text'])
		tokenizer = nltk.RegexpTokenizer(r"\w+")
		tokens = tokenizer.tokenize(string1) 

		docs[doc_id, each_doc['category']] = tokens

		for each_term in tokens:
			#saving the document in a dictionary
			if each_term not in terms:
				terms[each_term] = []
				terms[each_term].append((doc_id,each_doc['category']))
			else:
				if (doc_id,each_doc['category']) not in terms[each_term]:
					terms[each_term].append((doc_id,each_doc['category']))

		if each_doc['category'] not in category:
			category[each_doc['category']] = each_doc['category']

	terms['all_categories'] = category
	terms['sum_docs'] = doc_id
	terms['docs'] = docs
	
	return terms 

def main():
	try:
		path_vector_file = sys.argv[1] #path to  vector file
		path_json = sys.argv[3] #path to json test file
	except IndexError:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()
	try:
		file = open(path_json) 
	except FileNotFoundError:
		print("The path to the json file is invalid!")
		print("Exiting program...")
		sys.exit()
	try:
		f = open(path_vector_file) 
	except FileNotFoundError:
		print("The path to the vector file is invalid!")
		print("Exiting program...")
		sys.exit()

	if len(sys.argv) != 4:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()
	data = json.load(file)
	k = sys.argv[2]
	k = int(sys.argv[2])

	#tokenize and preprocess the training data
	terms = tokenize_function(data)

	#classes is a dictionary of all the classes in the document
	classes = terms['all_categories']
	sum_class = len(classes)
	
	#N is the total number of documents in the corpus
	N= terms['sum_docs'] 

	#Getting all documents with their tokens
	#key: docId,class
	#value: all terms in the document
	docs = terms['docs']

	#deleting 'all_categories', sum_docs and docs dictionary form terms
	terms.pop('all_categories', None) 
	terms.pop('sum_docs', None)
	terms.pop('docs', None)
	
	#calculating idf of each term, storing in dicitonary where key=term: value=idf
	idf_terms = get_idf(terms, N) 

	#calculating vector weights for each word for every doc in the testing data
	vector = get_vector(classes, terms, idf_terms, docs)

	#knn_model dictionary representing the vectors from the training data
	#key = (doc_id, class)
	#value = {vector for the document}
	knn_model = read_training_data(path_vector_file) 

	#converting string representation of dictionary to dictionary
	for each_doc in knn_model:
		knn_model[each_doc] = ast.literal_eval(knn_model[each_doc])

	#compute nearest neighbours
	#score is a dictionary with key: doc_id and its value is the score for the doc_id related to the test document
	score = {}
	#each_doc_nearest_neighbours has key every document, and value the top k scores doc_ids from the training data
	each_doc_nearest_neighbours = {}
	for each_doc in vector:
		for each_doc_in_training_data in knn_model:
			score[each_doc_in_training_data[1]] = 0
			for each_term_in_training_data in knn_model[each_doc_in_training_data]:
				#calculating euclidian distance
				if each_term_in_training_data in vector[each_doc]: #if term exists in both training and test data
					score[each_doc_in_training_data[1]] += (knn_model[each_doc_in_training_data][each_term_in_training_data] - vector[each_doc][each_term_in_training_data]) * (knn_model[each_doc_in_training_data][each_term_in_training_data] - vector[each_doc][each_term_in_training_data])
				elif each_term_in_training_data not in vector[each_doc]: #if term in training does not exist in test
					score[each_doc_in_training_data[1]] += (knn_model[each_doc_in_training_data][each_term_in_training_data]) * (knn_model[each_doc_in_training_data][each_term_in_training_data])
			
			for each_term_in_test_data in vector[each_doc]:#if term exist in test document but not in training document
				if each_term_in_test_data not in knn_model[each_doc_in_training_data]:
					score[each_doc_in_training_data[1]] += (vector[each_doc][each_term_in_test_data]) * (vector[each_doc][each_term_in_test_data])
			#square root
			score[each_doc_in_training_data[1]] = math.pow(score[each_doc_in_training_data[1]], (1/2)) #square root

		score_least_k = sorted(score, key=score.get)[:k]
		each_doc_nearest_neighbours[each_doc] = score_least_k
		
	#classifying each document
	#classify_docs stores each document_id and class for all documents in the test json
	#the value is the top k classes from knn using the training data
	classify_docs = {}
	for each_doc in each_doc_nearest_neighbours:
		classify_docs[each_doc] = []
		for each_close_doc in each_doc_nearest_neighbours[each_doc]: #for each document in the closest k documents
			for each_document_in_training_data in knn_model:
					if each_document_in_training_data[1] == each_close_doc: #doc_id
						classify_docs[each_doc].append(each_document_in_training_data[0])

	#assignment probabilities to each class
	#probability_class[each_doc] contains a dictionary with key each document in the test set
	#with value the number of classes the document corresponds to in the training data
	probability_class = {}
	for each_doc in classify_docs:
		probability_class[each_doc] = {}
		for each_class in classify_docs[each_doc]:
			for each_class_training_data in classes:
				if each_class == each_class_training_data:
					if each_class not in probability_class[each_doc]:
						probability_class[each_doc][each_class] = 1
					else:
						probability_class[each_doc][each_class] += 1

	#predicting the class for each document in the test data
	#prediction contains key (each_docid, actual class)
	#value = predicted class
	prediction={}
	for each_doc in probability_class:
		max_class = max(probability_class[each_doc], key=probability_class[each_doc].get)
		prediction[each_doc] = max_class

	#positives_negatives is a dictionary with key: class, TP/TN/FP/FN; value: the number of TP/TN/FP/FN
	positives_negatives = calculate_statistics(classes, prediction)

	#calculate values of precision, recall, f1, micro-average, macro-average and print to STDOUT
	print_statistics(positives_negatives, classes)

	#close vector fiel
	f.close()
	#close json file
	file.close()
	return 0
	
main()
