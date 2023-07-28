import json	
import nltk
import string
import csv	
import sys
import math

#Function to create the output file and test for errors
def intialise_file(fieldnames, path_store_document_vectors):
	try:
	    f = open(path_store_document_vectors)
	    #confirming to overwrite if model file already exist
	    print("Do you want to overwrite the existing file?('yes' or 'no' ONLY)")
	    decision = input("If 'no' the program will exit: " )
	    f.close()
	    if decision.lower() == 'yes':
	    	print("Overwriting file...")
	    else:
	    	print("Exiting the program..")
	    	sys.exit()
	except IOError:
	    print("Creating output file...")

	with open(path_store_document_vectors, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
		writer.writeheader()
	return 0


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

#Function to get the idf values for each term
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


#Function to write the idf value of each term in the output file
def write_output_idf(idf_terms, path_store_document_vectors, fieldnames):
	for each_term in idf_terms:
		with open(path_store_document_vectors, 'a', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
			writer.writerow({'idf/vector':'idf', 'term(t)':each_term, 'idf_value':idf_terms[each_term]})

	return 0

#Function to get the vector for each documents
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

#Function to write vector in file
def write_output_vector(vector, path_store_document_vectors, fieldnames):
	for each_doc in vector:
		with open(path_store_document_vectors, 'a', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
			writer.writerow({'idf/vector':'vector', 'class(c)': each_doc[1], 'vector':vector[each_doc]})

	return 0


def main():
	try: 
		path_json = sys.argv[1] #path to the json file with training data
		file = open(path_json)
	except FileNotFoundError:
		print("The path to the training data is invalid!")
		print("Exiting program...")
		sys.exit()
	try:
		path_store_document_vectors = sys.argv[2] #path to file to store the document vectors
	except IndexError:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()
	data = json.load(file)
	fieldnames = ['idf/vector','term(t)','idf_value', 'class(c)','vector']

	#create and initialise document vector file
	intialise_file(fieldnames, path_store_document_vectors) 

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

	#writing to file for idf
	write_output_idf(idf_terms, path_store_document_vectors, fieldnames)

	#calculating vector weights for each word for every document
	vector = get_vector(classes, terms, idf_terms, docs)

	#writing to file for vector
	write_output_vector(vector, path_store_document_vectors, fieldnames)

	#close json training file
	file.close()

	return 0
	
main()
