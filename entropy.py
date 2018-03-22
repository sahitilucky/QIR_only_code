import math
def entropy(doc_tf_idf):
	doc_list = doc_tf_idf.transpose().tolist()[0]
	doc_list = map(lambda l : l**2, doc_list)
	print sum(doc_list)
	entropy = 0
	for d in doc_list:
		if d!=0:
			entropy += d*math.log(d)
	#Boltzmann contant don't know now.  	
	return -entropy


def query_entropy(doc_tf_idf, q_vector):
	p = cosine_similarity(q_vector.transpose(),doc_tf_idf)**2
	p_ = 1-p
	doc_list =[p,p_] 
	entropy = 0
	printdoc_list
	for d in doc_list:
		if d!=0:
			entropy += d*math.log(d)
	#Boltzmann contant don't know now.  	
	return -entropy

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

docs_vectors = map(lambda l : (l[0],doc_tf_idf(l[1])), docs) 
print "doc_vectors_done"
print "with respect to random query"
all_term_query = normalize(np.matrix(np.ones(docs_vectors[0][1].shape)))
probs = map(lambda l: (l[0],cosine_similarity(all_term_query.transpose(),l[1])), docs_vectors)
sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
sorted_prob = sorted_prob[:]
entropies1 = map(lambda l : (l[0], query_entropy(docs_vectors[l[0]-1][1], all_term_query), l[1]) , sorted_prob)
for e in entropies1:
	print e

print "with respect to query"
q_vector = doc_tf_idf(queries[0][1])
probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
sorted_prob = sorted_prob[:]
entropies = map(lambda l : (l[0], query_entropy(docs_vectors[l[0]-1][1], q_vector), l[1]) , sorted_prob)
for e in entropies:
	e1 = filter(lambda l : l[0] == e[0],entropies1)[0]
	change_in_entropy = (e[1]-e1[1])
	print (e[0],e[1],e[2],e1[1],change_in_entropy)


print "with respect to query"
q_vector = doc_tf_idf(queries[0][1])
probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
sorted_prob = sorted_prob[:]
entropies = map(lambda l : (l[0], query_entropy(docs_vectors[l[0]-1][1], q_vector), l[1]) , sorted_prob)
for e in entropies:
	e1 = filter(lambda l : l[0] == e[0],entropies1)[0]
	change_in_entropy = (e[1]-e1[1])
	print (e[0],e[1],e[2],e1[1],change_in_entropy)


