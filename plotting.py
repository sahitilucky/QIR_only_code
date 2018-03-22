from common import *
import pickle
import scipy
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold

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
'''
idf = sorted(idf.items(), key = lambda l: l[1], reverse = True)
print len(idf)
for item in idf:
	print item
'''
from Document_rep import *
from Query_rep import *
idf = sorted(idf.items(), key = lambda l: l[1], reverse = True)
#for a group of words
'''
idf = sorted(idf.items(), key = lambda l: l[1], reverse = True)
test_words = idf[2500:2600]
for w in test_words:
	print w
	dens_op = single_term_window(preprocessing_nltk(w[0])[0], docs)
'''
def dump_docoperators(start,end):
	for i in range(start,end):
		doc_operator = divide_fragments(docs[i][1])
		sparse_matrix = scipy.sparse.csc_matrix(doc_operator)
		scipy.sparse.save_npz('../../cran/doc_operators/sparse_matrix'+str(i+1)+'.npz', sparse_matrix)


#words closer to a density operator using a term space
def closer_words(term):
	query = preprocessing_nltk(term)
	sample_docs = idf[:1000]
	density_op = multi_term_mixture(query, docs)
	print "density operator done"
	rels = []
	for i in range(len(allwords)):
		rels += [(allwords[i],density_op[(i,i)])]
	prob = sorted(rels, key = lambda l:l[1],reverse = True)
	return prob[:100]

def rel_document_sample_docs(query):
	print query
	sample_docs = get_rel_irrel_docs(qrels[query[0]],doc_operators)
	print map(lambda l: l[0], sample_docs)	
	density_op = multi_term_mixture(query[1], docs)
	print "density operator done"
	density_op = scipy.sparse.csc_matrix(density_op)
	start = time.time()
	prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),sample_docs)
	prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	end = time.time()
	print prob[:50]
	print (end-start)
	return prob[:50]

def rel_document_single_term(term, sample_docs):
	#qrels = read_query_rels('../../cran/cranqrel')
	#sample_docs = get_rel_irrel_docs(qrels[query[0]],doc_operators)
	density_op = multi_term_mixture([term], docs)
	print "density operator done"
	density_op = scipy.sparse.csc_matrix(density_op)
	start = time.time()
	prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),sample_docs)
	prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	end = time.time()
	#print map(lambda l: l[0] ,sample_docs)
	print prob
	print (end-start)
	return prob
'''
target = open("many_queries_APS.txt","a")
APS =[]
doc_operators = load_sparce_mats(1,1400)
print "docs_operators done"
for i in range(1):
	prob = rel_document_sample_docs(queries[i])
	AP1 = calculate_AP(qrels[queries[i][0]],prob[:30])
	target.write(str(prob)+'\n')
	target.write(str(AP1)+'\n')
	print AP1
	APS += [AP1]
print calculate_MAP(APS)
target.flush()
target.close()
'''
def idf_probabilities(density_ops,d,num):
	probs = []
	for i in range(len(density_ops)):
		B = density_ops[i][0]	
		probs += [prob_projection(B,d)]
	print ("over "+str(num))
	return (probs)


target = open("analysis1.txt","a")
target4 = open("analysis2.txt","a")
docs_vectors = map(lambda l : (l[0],doc_tf_idf(l[1])), docs) 
print "doc_vectors_done"

'''
target = open("tf_idf_results.txt","a")
for i in range(30, 50):
	print i 	
	q_vector = doc_tf_idf(queries[i][1])
	target.write('Query '+ str(queries[i][0])+'\n')
	print q_vector.sum()
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])),docs_vectors)
	print "probs done"
	sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]	
	target.write(str(sorted_prob[:50])+'\n')
	target.write(str(AP1)+'\n')
target.flush()
target.close()
'''

#update document vector
i=10
doc_num = 27
print "Query " + str(i+1)
print "Doc " + str(doc_num+1)
q_vector = doc_tf_idf(queries[i][1])
query_subspace = q_vector*q_vector.transpose()
print docs_vectors[doc_num][0]
required_doc = filter(lambda l : l[0] == (doc_num+1), docs_vectors)[0][1]
find_near_vectors(required_doc, docs_vectors)
(nghbr_vectors,updated_doc) = doc_projection_update(query_subspace, required_doc, docs_vectors)
for n in nghbr_vectors:
	print docs_vectors[n[0]-1][0]
	docs_vectors[n[0]-1] = n
	#analyse([1],n[1])
docs_vectors[doc_num] = (docs_vectors[doc_num][0], updated_doc)
find_near_vectors(docs_vectors[doc_num][1] , docs_vectors)
print docs_vectors[doc_num][1]
#analyse([1],docs_vectors[doc_num][1])
probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), docs_vectors)
print "probs done"
sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob[:30])
print AP1
print sorted_prob[:50]	


import numpy as np
from sklearn.manifold import TSNE
print docs_vectors[0][1].transpose()[0,:][0]
doc_ops = map(lambda l : l[1].transpose()[0,:][0], docs_vectors) 
print doc_ops
doc_ops = np.matrix(doc_ops)
print doc_ops.shape
X = doc_ops
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
print Y.shape
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE projection")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.show()

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



