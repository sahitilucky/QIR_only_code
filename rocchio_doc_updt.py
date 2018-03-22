from common import *
import pickle
import scipy
import time
#docs = read_data_docs('../../cran/cran.all.1400', 1400)
#print docs[:3]
#pickle.dump(docs, open("../../cran/cran_1400_2.pkl", "wb"))
docs = pickle.load(open("../../cran/cran_1400_2.pkl" ,"rb"))
queries = read_queries('../../cran/cran.qry')
qrels = read_query_rels('../../cran/cranqrel')
print "--------Queries-------"
#print queries[:5]
#pickle.dump(docs, open("../../cran/query_500.pkl", "wb") )
#(idf,allwords2) = get_all_terms(docs,queries)
#pickle.dump(idf, open("../../cran/IDF_1400_2.pkl", "wb"))
#pickle.dump(allwords2, open("../../cran/allwords_1400_2.pkl", "wb"))
idf = pickle.load(open("../../cran/IDF_1400_2.pkl", "rb"))
allwords2 = pickle.load( open("../../cran/allwords_1400_2.pkl", "rb"))
set_params(allwords2,idf)
print len(allwords2)

from Document_rep import *
from Query_rep import *
idf = sorted(idf.items(), key = lambda l: l[1], reverse = True)
#for a group of words

docs_vectors = map(lambda l : (l[0],doc_tf_idf(l[1])), docs) 
print "doc_vectors_done"


def get_doc_num(query, q_vector, docs_vectors):
	qs = qrels[query[0]]
	qs = map(lambda l: l[0], qs)
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]
	i = 0	
	sorted_prob = sorted_prob[:30]
	doc_nums = [] 	
	for s in sorted_prob:
		if s[0] in qs:
			doc_nums += [s[0]-1]
			i = i+1
			if(i == 2):			
				break
	return doc_nums


def get_doc_num2(query, q_vector, docs_vectors, doc_num):
	qs = qrels[query[0]]
	qs = map(lambda l: l[0], qs)
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]
	i = 0	
	sorted_prob = sorted_prob[:30]
	doc_nums = [] 	
	for s in sorted_prob:
		if s[0] in qs:
			doc_nums += [s[0]-1]
			i = i+1
			if(i == 2):			
				break
	if doc_num in doc_nums:
		doc_nums.remove(doc_num)	
	return doc_nums		

#update document vector
qs = range(42,43)
for i in qs:
	print "Query " + str(i+1)
	q_vector = doc_tf_idf(queries[i][1])
	doc_nums = get_doc_num(queries[i], q_vector, docs_vectors)
	print "Doc " + str(doc_nums)
	#find_near_vectors(docs_vectors[doc_nums[0]][1], docs_vectors)
	#find_near_vectors(docs_vectors[doc_nums[1]][1], docs_vectors)
	query_subspace = q_vector*q_vector.transpose()
	new_doc_vectors = docs_vectors[:]
	#print docs_vectors[doc_num][0]
	doc_num1 = 0 	
	if (len(doc_nums)>0):
		doc_num1 = doc_nums[0]
		required_doc = filter(lambda l : l[0] == (doc_num1+1), docs_vectors)[0][1]
		#find_near_vectors(required_doc, docs_vectors)
		(nghbr_vectors,updated_doc1,updated_query1) = query_doc_projection_update(required_doc, docs_vectors, q_vector,1,queries[i][1])
		q_vector = updated_query1		
		for n in nghbr_vectors:
			#print docs_vectors[n[0]-1][0]
			new_doc_vectors[n[0]-1] = n
			#analyse([1],n[1])
		new_doc_vectors[doc_num1] = (new_doc_vectors[doc_num1][0], updated_doc1)
		q_vector = updated_query1
	
	doc_nums = get_doc_num2(queries[i], q_vector, new_doc_vectors,doc_num1)
	print "Doc " + str(doc_nums)
	if (len(doc_nums)>0):
		#find_near_vectors(docs_vectors[doc_nums[0]][1], docs_vectors)
		doc_num = doc_nums[0]
		doc_num = 38		
		required_doc = filter(lambda l : l[0] == (doc_num+1), new_doc_vectors)[0][1]
		(nghbr_vectors, updated_doc, updated_query) = query_doc_projection_update(required_doc, new_doc_vectors, q_vector,1,queries[i][1])
		for n in nghbr_vectors:
			#print docs_vectors[n[0]-1][0]
			new_doc_vectors[n[0]-1] = n
			#analyse([1],n[1])
		new_doc_vectors[doc_num] = (new_doc_vectors[doc_num][0], updated_doc)
		q_vector = updated_query
	new_doc_vectors[doc_num1] = (new_doc_vectors[doc_num1][0], updated_doc1)
	
	#find_near_vectors(new_doc_vectors[doc_num][1] , new_doc_vectors)
	#analyse([1],docs_vectors[doc_num][1])
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), new_doc_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]	

'''
import numpy
for i in range(10):
	similarities1 = find_near_vectors_threshold(docs_vectors[i][1], docs_vectors)
	std = numpy.std(similarities1)
	print std
	avg = float(sum(similarities1))/float(len(similarities1))	
	print avg
	print (avg-std)
	histogram = {}
	for s in similarities1:
		s = round(s,2)
		if s not in histogram:
			histogram[s] = 1
		else:
			histogram[s] += 1
	histogram = histogram.items()
	histogram = sorted(histogram, key = lambda l :l[0], reverse = True)
	print histogram 
'''

'''
#rocchi update
from rocchio import *

for i in range(50):
	query = queries[i]
	print "Query " + str(i+1)
	q_vector = doc_tf_idf(query[1])
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]
	(rel_docs, non_rel_docs) = split_rel_non_rel_docs(query, sorted_prob[:30], qrels)
	rel_docs = map(lambda l : docs_vectors[l[0]-1], rel_docs)
	non_rel_docs = map(lambda l : docs_vectors[l[0]-1], non_rel_docs)
	if(len(rel_docs) == 0):
		rel_docs = [(0, np.zeros(non_rel_docs[0][1].shape))]
	num_rel_docs = map(lambda l : l[0], rel_docs)	
	print "Docs " + str(num_rel_docs) + " " + str(non_rel_docs[0][0])	
	q_vector = normalize(rocchio(q_vector, rel_docs, non_rel_docs))
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1], reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]
	
	(rel_docs, non_rel_docs) = split_rel_non_rel_docs(query, sorted_prob[1:30], qrels)
	rel_docs = map(lambda l : docs_vectors[l[0]-1], rel_docs)
	non_rel_docs = map(lambda l : docs_vectors[l[0]-1], non_rel_docs)
	if(len(rel_docs) == 0):
		rel_docs = [(0, np.zeros(non_rel_docs[0][1].shape))]
	num_rel_docs = map(lambda l : l[0], rel_docs)	
	print "Docs " + str(num_rel_docs) + " " + str(non_rel_docs[0][0])	
	q_vector = normalize(rocchio(q_vector, rel_docs, non_rel_docs))
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1], reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]
'''

'''
#update query vector
i=29
doc_num = 512
q_vector = doc_tf_idf(queries[i][1])
query_subspace = q_vector*q_vector.transpose()
doc_subspace = docs_vectors[doc_num][1]*docs_vectors[doc_num][1].transpose()
print docs_vectors[doc_num][0]
q_vector = doc_subspace*q_vector
q_vector = normalize(q_vector)
probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
print "probs done"
sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob[:30])
print AP1
print sorted_prob[:50]	
'''

'''
query = queries[10]
density_ops = tensor_product_T2(query[1], docs)
i=0
for d in density_ops:
	print query[1][i]
	densityop = d[0]
	densityop = densityop.transpose().tolist()
	(eigw, eigv) = eigenvaldecomp(densityop, 5)
	analyse(eigw, eigv)
	i+=1
idfs = map(lambda l : l[1],density_ops)
print "Query subspace"
get_query_subspace(density_ops)
'''
















