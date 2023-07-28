import json	
import nltk
import string
import csv	
import sys
import math

#Function to initialise index file each time the program is run:
def intialise_file(fieldnames, path_store_model):
	try:
	    f = open(path_store_model)
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
	    print("Creating model file...")

	with open(path_store_model, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
		writer.writeheader()
		
#Function for pre-processing the data
def tokenize_function(data, keys):
	distinct_terms = {} #dictionary containt distinct terms in the vocabulary
	count = 0 #count number of distinct terms
	word_with_class = {} #dictionary containing a distinct term in a distinct class
	all_categories = {} #dictionary containing all distinct classes
	sum_documents = 0 #sum of documents in the file
	
	#Looping through all the data
	for each_doc in data:
		sum_documents += 1

		#tokenization/pre-processing
		string1 = str(each_doc['text'])
		tokenizer = nltk.RegexpTokenizer(r"\w+")
		tokens = tokenizer.tokenize(string1) 

		if each_doc['category'] not in all_categories:
			#initialising the number of documents in the class to be 0
			all_categories[each_doc['category']]=0
			#initialising number of words in the class to zero
			word_with_class[each_doc['category'], 'total'] = 0

		all_categories[each_doc['category']]= all_categories[each_doc['category']] + 1

		#Looping through each term in the document
		#to calculate number of distinct terms in the vocabulary and create word_with_class dictionary
		for each_word in tokens:
			if each_word not in distinct_terms: 
				distinct_terms[each_word]=each_word
				count+=1
			if (each_doc['category'], each_word) not in word_with_class:
				word_with_class[each_doc['category'], each_word] = 0
				
			#add one to the value of the (class, term) everytime the term appears again in the same class
			word_with_class[each_doc['category'], each_word] = word_with_class[each_doc['category'], each_word] + 1 

			#add one for each time a word appears in a class
			#word_with_class[each_doc['category'], total] is the sum of words in that class
			word_with_class[each_doc['category'], 'total'] = word_with_class[each_doc['category'], 'total'] + 1

	#Adding every term into every class:
	for each_term in distinct_terms:
		for each_class in all_categories:
			if (each_class, each_term) not in word_with_class:
				word_with_class[each_class, each_term] = 0
	
	word_with_class['Count'] = count #word_with_class['Count'] contains number of distinct terms in the vocabulary

	#creating an array in the dictionary word_with_class to store all classes and the number of docs in each class
	word_with_class['all_categories']=all_categories

	#sum of all documents
	word_with_class['sum_documents'] = sum_documents
	
	return word_with_class 
	

def main():
	try: #if path to store model not included in command line
		try: 
			path_json = sys.argv[1] #path to the json file
			file = open(path_json) 
		except FileNotFoundError:
			print("The path to the training data is invalid!")
			print("Exiting program...")
			sys.exit()
		path_store_model = sys.argv[2] #path to store model
	except IndexError:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()

	if len(sys.argv) != 3:
		print("The program has been called with the wrong number of arguments!")
		print("Exiting program...")
		sys.exit()

	data = json.load(file)
	#KEYS HARDCODED, ASSUMING THAT ONLY 2 ZONES NAMED "TEXT" AND "CATEGORY" WILL BE CONTAINED IN JSON FILE FOR EVALUATION, SIMILAR TO JSON FILE GIVEN
	keys=["text","category"]
	fieldnames = ['likelihood/prior','Class(c)','Term(t)', 'P(c)','P(t|c)']

	#create and initialise bbc model tsv file
	intialise_file(fieldnames, path_store_model) 
	
	#word_with_class is a dictionary: distinct (class, term) key and the number of times the term in that class appears as the value
	word_with_class = tokenize_function(data, keys)

	#writing prior to file
	for each_class in word_with_class['all_categories']:
		#calculating prior
		#total number of documents = word_with_class['sum_documents']
		#number of documents in the class = word_with_class['all_categories'][each_class]
		with open(path_store_model, 'a', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
			writer.writerow({'likelihood/prior':'prior', 'Class(c)':each_class, 'P(c)':(word_with_class['all_categories'][each_class])/word_with_class['sum_documents']})

	#writing likelihood to file
	for each_key in word_with_class.keys():
		if each_key != 'Count' and each_key != 'all_categories' and each_key!= 'sum_documents':
			if each_key[1] != 'total':
				with open(path_store_model, 'a', newline='') as csvfile:
					writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
					#For calculating P(t|c):
					#Numerator: T(ct) + 1 = word_with_class[each_key] + 1
					#Denominator: B = word_with_class['Count'], Total number of terms in that class = word_with_class[each_key[0],'total']
					prob = (word_with_class[each_key]+1)/((word_with_class['Count'])+word_with_class[each_key[0],'total'])
					writer.writerow({'likelihood/prior':'likelihood', 'Class(c)':each_key[0], 'Term(t)': each_key[1], 'P(t|c)':prob})

	#close json file
	file.close()
	return 0
	
main()
