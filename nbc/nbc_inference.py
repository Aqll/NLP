import json	
import nltk
import string
import csv	
import sys
import math

#Function to read the conditional probability
def get_condprob(path_store_model):
	cond_prob = {} #dictionary containing term,class and it's conditional probability p(t|c)
	with open(path_store_model, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = '\t')
		for row in reader:
			if row['likelihood/prior'] == 'likelihood':
				cond_prob[row['Term(t)'], row['Class(c)']] = row['P(t|c)']

	return cond_prob


#Function to read the prior
def get_prior(path_store_model):

	prior = {} #dictionary containing class and it's prior
	with open(path_store_model, 'r', newline='') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = '\t')
		for row in reader:
			if row['likelihood/prior'] == 'prior':
				prior[row['Class(c)']] = row['P(c)']
			else:
				#since priors always at the top of the file since they are written first by my nbc_train.py program, can stop reading after last prior read
				return prior

#Function for pre-processing the data
def tokenize_function(data):
	terms = {} #dictionary containing all terms found in a document
	category={} #list containing all categories
	doc_id = 0 #used to identify a document

	#Looping through all the data
	for each_doc in data:
		#tokenization/pre-processing
		string1 = str(each_doc['text'])
		tokenizer = nltk.RegexpTokenizer(r"\w+")
		tokens = tokenizer.tokenize(string1) 

		#saving the document in a dictionary
		terms[doc_id,each_doc['category']] = tokens

		if each_doc['category'] not in category:
			category[each_doc['category']] = each_doc['category']
		doc_id +=1

	terms['all_categories'] = category
	
	return terms 

#Function to calculate log of each term in a dictionary
def calculate_log(dicitonary):
	for each in dicitonary:
		dicitonary[each] = math.log(float(dicitonary[each]),10)

	return dicitonary


def do_prediction(docs_with_terms, prior, cond_probability):
	predicted_class_each_doc = {} #dictionary to store each document and the predicted class
	score = {} #score dictionary which stores prediction for each class for a document

	for each_doc in docs_with_terms: #each_doc is the (doc_id,class)
		#initialise score for each class for the document according to the prior
		for each_class in docs_with_terms['all_categories']:
			score[each_class] = prior[each_class]
	
		if each_doc != 'all_categories': #docs_with_terms[each_doc] are the terms in the document. 
			for each_term in docs_with_terms[each_doc]:
				for each_class in docs_with_terms['all_categories']:

					#if the term does not appear in the index file(trained data)
					if (each_term, each_class) not in cond_probability:
						cond_probability[(each_term, each_class)] = 0
					score[each_class] += cond_probability[(each_term, each_class)]
			
			predicted_class =  max(score, key=score.get) #to get the class with highest score
			
			predicted_class_each_doc[each_doc] = predicted_class

	return predicted_class_each_doc

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
def print_statistics(docs_with_terms, positives_negatives):

	#intialise sum of F1, TP, FP, FN of all classes and number of classes to zero
	sum_F1 = sum_TP = sum_FP = sum_FN = num_class = 0 
	for each_class in docs_with_terms['all_categories']:
		num_class +=1
		TP = positives_negatives[each_class, 'true_positive']
		FP = positives_negatives[each_class, 'false_positives']
		FN = positives_negatives[each_class, 'false_negatives']
		TN = positives_negatives[each_class, 'true_negatives']

		precision = TP/(TP+FP)
		recall = TP/(TP+FN)
		F1 = (2*precision*recall)/(precision+recall)

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


def main():
	try:
		path_store_model = sys.argv[1] #path model already created
		path_json = sys.argv[2] #path to json test file
	except IndexError:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()
	file = open(path_json) 
	data = json.load(file)
	
	if len(sys.argv) != 3:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()

	#docs_with_terms is a dictionary which contains each document and all the terms that appear in that document
	#docs_with_terms: key=(doc_id,class), value=all terms appearing in that document
	docs_with_terms = tokenize_function(data)

	#classes is a dictionary of all the classes in the document
	classes = docs_with_terms['all_categories'] 
	
	#prior is a dictionary with key: class; and value: prior of the class
	prior = get_prior(path_store_model)
	
	#converting prior to log prior
	prior = calculate_log(prior)

	#cond_probability is a dictionary with key: term,class; and value: P(t|c)
	cond_probability = get_condprob(path_store_model)

	#converting cond_probability to log cond_probability
	cond_probability = calculate_log(cond_probability)

	#predicted_class_each_doc is a dictionary with key: (doc_id,class); and value predicted_class
	predicted_class_each_doc = do_prediction(docs_with_terms, prior, cond_probability)

	#positives_negatives is a dictionary with key: class, TP/TN/FP/FN; value: the number of TP/TN/FP/FN
	positives_negatives = calculate_statistics(classes, predicted_class_each_doc)
	
	#calculate values of precision, recall, f1, micro-average, macro-average and print to STDOUT
	print_statistics(docs_with_terms, positives_negatives)
	
	
	#close json file
	file.close()
	return 0
	
main()
