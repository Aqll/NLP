import json	
import nltk
import string
import csv	
import sys
import math
from nltk.corpus import stopwords
from collections import Counter

#Function for pre-processing the data
def tokenize_function(data):
	terms = {} #dictionary containing all terms found in a document
	category={} #dictionary containing all classes and the total number of documents occuring in that class
	doc_id = 0 #used to identify a document
	#I removed stopword because these term do not provide information about the class
	stop_words = stopwords.words('english')
	stopwords_dict = Counter(stop_words) #using dictionary to run faster than using list 

	#Looping through all the data
	for each_doc in data:
		#tokenization/pre-processing
		string1 = str(each_doc['text'])
		tokenizer = nltk.RegexpTokenizer(r"\w+")
		tokens = tokenizer.tokenize(string1) 
		tokens = [word for word in tokens if not word in stopwords_dict]

		#terms is a dictionary of all distinct terms. key: term, value = dictionary explained below
		#terms[each_word] is dictionary which contains the classes and the number of time the term 'each_word' appears in that class
		#key: each_word, value:dictionary explained below
		#terms[each_word][each_doc['category']] is a dictionary which contains all the documents for which the term appears in the class
		#key: class, value: documents with term each_word in the class 
		for each_word in tokens:
			if each_word not in terms: 
				terms[each_word] = {}
				
		for each_word in tokens:
			if each_doc['category'] not in terms[each_word]:
				terms[each_word][each_doc['category']] = {}
				terms[each_word][each_doc['category']][doc_id] = 1
			else: 
				terms[each_word][each_doc['category']][doc_id] = 1

		if each_doc['category'] not in category:
			category[each_doc['category']] = 1
		else:
			category[each_doc['category']] += 1

		doc_id +=1

	for each_term in terms:
		for each_class in terms[each_term]:
			terms[each_term][each_class] = len(terms[each_term][each_class])

	for each_term in terms:
		total = 0 #total is number of documents the term appears in all classes
		for each_class in terms[each_term]:
			total += terms[each_term][each_class]
		terms[each_term]['total'] = total

	terms['all_categories'] = category
	terms['sum_docs'] = doc_id

	return terms 

#Function to calculate the mutual information score for each term in a class
def get_score(t, c,N11, N10, N01, N00, N):
	
	value = 0
	product = N11/N 
	if product != 0: #if no document in the class contains word t
		log_numerator = N * N11
		log_denominator = (N11 + N10) * (N11 + N01)
		value = product*math.log((log_numerator/log_denominator), 2)
	else:
		value = 0 #Assigned zero value because the term does not contribute in helping whether a doc is in a class

	product = N10/N 
	if product != 0: #if no other classes contain word t
		log_numerator = N * N10
		log_denominator = (N11 + N10) * (N10 + N00)	
		value += product*math.log((log_numerator/log_denominator), 2)
	else:
		value += 0

	product = N01/N
	log_numerator = N * N01
	log_denominator = (N00 + N01) * (N01 + N11)
	value += product*math.log((log_numerator/log_denominator), 2)

	product = N00/N
	log_numerator = N * N00
	log_denominator = (N00 + N01) * (N00 + N10)
	value += product*math.log((log_numerator/log_denominator), 2)
	
	return value

def main():
	try: 
		path_json = sys.argv[1] #path to the json file
		file = open(path_json)
	except FileNotFoundError:
		print("The path to the training data is invalid!")
		print("Exiting program...")
		sys.exit()
	try:
		k = sys.argv[2]
		path_store_filtered_file = sys.argv[3] #path to store the filtered training file
	except IndexError:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()

	
	try:
	    f = open(path_store_filtered_file)
	    #confirming to overwrite if model file already exist
	    print("Do you want to overwrite the existing file?('yes' or 'no' ONLY)")
	    decision = input("If 'no' the program will exit: " )
	    f.close()
	    if decision.lower() == 'yes':
	    	print("Overwriting output file...")
	    else:
	    	print("Exiting the program..")
	    	sys.exit()
	except IOError:
	    print("Creating output file...")

	f = open(path_store_filtered_file, "w")
	data = json.load(file)
	value = [] #value is a list which stores the scores for each word

	if len(sys.argv) != 4:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()

	k = int(sys.argv[2])

	#KEYS HARDCODED, ASSUMING THAT ONLY 2 ZONES NAMED "TEXT" AND "CATEGORY" WILL BE CONTAINED IN JSON FILE FOR EVALUATION, SIMILAR TO JSON FILE GIVEN
	keys=["text","category"]
	fieldnames = ['likelihood/prior','Class(c)','Term(t)', 'P(c)','P(t|c)']

	#docs_with_terms is a dictionary which contains each document and all the terms that appear in that document
	#docs_with_terms: key=(term, docid), value=the class to which the documents belong
	docs_with_terms = tokenize_function(data)

	#classes is a dictionary of all the classes in the document
	classes = docs_with_terms['all_categories']
	sum_class = len(classes)
	
	#N is the total number of documents in the corpus
	N = docs_with_terms['sum_docs'] 

	#deleting 'all_categories' and sum_docs
	docs_with_terms.pop('all_categories', None) 
	docs_with_terms.pop('sum_docs', None)

	##N11: number of documents in class c that constains term t
	#N10: number of documents not in class c that contains term t
	#N01: number of documents in class c that do not contain term t
	#N00: number of documents not in class c that do not contain term t
	N10 = 0
	N01 = 0
	N00 = 0
	
	f = open(path_store_filtered_file, "a")
	f.write("[\n    ")

	count_class = 0 #counter for class
	for each_class in classes: #for a given class c
		count_class +=1
		for each_term in docs_with_terms: #for a given term t
			N11 = docs_with_terms[each_term].get(each_class, 0)
			N10 = docs_with_terms[each_term].get('total') - N11
			N01 = classes[each_class] - N11
			N00 = N - (N01 + N10 + N11)
			
			value1=get_score(each_term, each_class, N11, N10, N01, N00, N)
			value.append((value1, each_term, each_class))

		#sorting the scores in ascending order
		value.sort()
		value = value[-k:] #get the top k scores
		text = ""
		counter = 0
		for each in value:
			counter +=1
			text = text + each[1]
			if counter != k:
				text = text + " " #concatonating the top words

		#formatting the output
		f.write("    {\n        \"category\": \""+each_class+"\",\n")
		f.write("        \"text\": \""+text+"\"")
		if count_class != sum_class:
			f.write("\n    },\n")

		#re-initalisation
		value = []
	f.write("\n    }\n]")

	#close json writing filtered file
	f.close()
	#close json training file
	file.close()
	return 0
	
main()
