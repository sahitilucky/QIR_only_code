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
'''
idf = sorted(idf.items(), key = lambda l: l[1], reverse = True)
print len(idf)
for item in idf:
	print item
'''
from Document_rep import *
from Query_rep import *

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
	idf = sorted(idf.items(), key = lambda l: l[1], reverse = True)
	query = preprocessing_nltk(term)
	sample_docs = idf[:1000]
	density_op = multi_term_mixture(query, docs)
	print "density operator done"
	rels = []
	for i in range(len(allwords)):
		rels += [(allwords[i],density_op[(i,i)])]
	sorted_prob = sorted(rels, key = lambda l:l[1],reverse = True)
	return sorted_prob[:100]

def rel_document_sample_docs(query):
	print query
	sample_docs = get_rel_irrel_docs(qrels[query[0]],doc_operators)
	print map(lambda l: l[0] ,sample_docs)	
	density_op = multi_term_mixture(query[1], docs)
	print "density operator done"
	density_op = scipy.sparse.csc_matrix(density_op)
	start = time.time()
	prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),sample_docs)
	sorted_prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	end = time.time()
	print sorted_prob[:50]
	print (end-start)
	return sorted_prob[:50]

def rel_document_single_term(term, sample_docs):
	#qrels = read_query_rels('../../cran/cranqrel')
	#sample_docs = get_rel_irrel_docs(qrels[query[0]],doc_operators)
	density_op = multi_term_mixture([term], docs)
	print "density operator done"
	density_op = scipy.sparse.csc_matrix(density_op)
	start = time.time()
	prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),sample_docs)
	sorted_prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	end = time.time()
	#print map(lambda l: l[0] ,sample_docs)
	print sorted_prob
	print (end-start)
	return sorted_prob
import ast
f = open("query_APS_T1.txt","r")
target = open("query_APS_T1_new.txt","w")
APS =[]
line = f.readline()
i = 0
while line:
	sorted_prob = ast.literal_eval(line)
	print sorted_prob
	x = i
	print x
	AP1 = calculate_AP(qrels[queries[x][0]],sorted_prob[:30])
	i = i+1	
	target.write(str(sorted_prob)+'\n')
	target.write(str(AP1)+'\n')
	print AP1
	APS += [AP1]
	line=f.readline()
	line=f.readline()
print calculate_MAP(APS)
target.flush()
target.close()
f.close()


'''
doc_operators = map(lambda l: (l[0],np.matrix(text_to_vector([l[0]]))),sample_docs)
m = doc_operators[0][1].shape[1]
new_doc_operators = []
for i in range(len(doc_operators)):
	print i
	ind = doc_operators[i][1].tolist()[0].index(1.0)
	doc_operator = np.matrix(np.zeros([m,m]))
	doc_operator[(ind,ind)] = 1  
	new_doc_operators += [(doc_operators[i][0], doc_operator)] 
print "docs_operators done"
prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),new_doc_operators)
sorted_prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
print sorted_prob
'''

'''
query = list(filter(lambda t : single_term_window(t, docs)[0]!=[] , query[1]))
print query
density_ops = map(lambda t : (t,IDF[t]*single_term_window(t, docs)[0]),query)

for d in sample_docs:
	print d
	doc_op = divide_fragments(d[1])
	prob_of_relevance = map(lambda l: (l[0] , prob_projection(l[1],doc_op)),density_ops)
	sorted_prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	print sorted_prob
print map(lambda t : IDF[t],query)	
'''

'''
multi_term_mixture(["experiment","aerodynamic"], docs)
mixture_of_superpositions(["experiment","slipstream"], docs)
tensor_product_T1(["experiment","aerodynamic"], docs)
tensor_product_T2(["experiment","aerodynamic"], docs)
'''
'''
qrels = read_query_rels('../../cran/cranqrel')
print qrels[1]
docs = read_data_docs('../../cran/cran.all.1400', 100)
doc_operators = map(lambda l: (l[0],divide_fragments(l[1])),docs[:10])
#pickle.dump(doc_operators, open("../../cran/doc_operators_50.pkl", "wb"))
'''
'''
APs = []
for i in range(10):
	#getting relevant docs using mixture method
	density_op = multi_term_mixture(queries[i][1], docs)
	print "density operator done"
	prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),doc_operators)
	sorted_prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	print sorted_prob[:10]
	print queries[i][0]
	AP = calculate_AP(qrels[queries[i][0]],sorted_prob[:10])
	APs += [AP]
	print AP
print calculate_MAP(APs)

#getting relevant docs using mixture of superpositions method
density_op,weight_set_vectors = mixture_of_superpositions(queries[0][1], docs)
prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),doc_operators)
sorted_prob = sorted(prob_of_relevance,key = lambda l:l[1], reverse = True)
print sorted_prob[:10]

APs = []
for i in range(10):
	#getting relevant docs using T2 method
	density_ops = tensor_product_T2(queries[i][1], docs)
	prob_of_relevance = map(lambda l: (l[0], projections_T2(density_ops,l[1])),doc_operators)
	sorted_prob = sorted(prob_of_relevance,key = lambda l:l[1],reverse = True)
	print sorted_prob[:10]
	AP = calculate_AP(qrels[queries[i][0]],sorted_prob[:10])
	APs += [AP]
	print AP
print calculate_MAP(APs)


#getting relevant docs using T1 method
density_ops = tensor_product_T1(queries[0][1], docs)
prob_of_relevance = map(lambda l: (l[0], projections_T1(density_ops,l[1])),doc_operators[:30])
sorted_prob = sorted(prob_of_relevance,key = lambda l:l[1],reverse = True)
print sorted_prob[:10]
calculate_AP(qrels[queries[0][0]],sorted_prob[:10])
'''
'''
global IDF
query = queries[0][1]
query = list(filter(lambda t : single_term_window(t, docs)[0]!=[] , query))
print query
density_op = map(lambda t : IDF[t]*single_term_window(t, docs)[0],query)
print doc_operators[1][0]
rels = map(lambda l : prob_projection(l,doc_operators[1][1]),density_op)
print rels
print doc_operators[2][0]
rels = map(lambda l : prob_projection(l,doc_operators[2][1]),density_op)
print rels
'''
'''
#getting relevant docs using T2 method
density_ops = tensor_product_T2(queries[0][1], docs)
prob_of_relevance = map(lambda l: (l[0], projections_T2(density_ops,l[1])),doc_operators)
sorted_prob = sorted(prob_of_relevance,key = lambda l:l[1],reverse = True)
print sorted_prob[:10]
'''












