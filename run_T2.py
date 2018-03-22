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


def rel_document_sample_docs(query):
	print query
	sample_docs = get_rel_irrel_docs(qrels[query[0]],doc_operators)
	print map(lambda l: l[0] ,sample_docs)	
	density_op =  tensor_product_T2(query[1], docs)
	print "density operator done"
	density_op = map(lambda l: (scipy.sparse.csc_matrix(l[0]),l[1]), density_op)
	start = time.time()
	prob_of_relevance = map(lambda l: (l[0], projections_T2(density_op,l[1])),sample_docs)
	sorted_prob1 = sorted(prob_of_relevance, key = lambda l:l[1][0],reverse = True)
	sorted_prob2 = sorted(prob_of_relevance, key = lambda l:l[1][1],reverse = True)
	end = time.time()
	print sorted_prob1[:50]
	print sorted_prob2[:50]
	print (end-start)
	return (sorted_prob1[:50],sorted_prob2[:50])

def rel_document_single_term(term, sample_docs):
	#qrels = read_query_rels('../../cran/cranqrel')
	#sample_docs = get_rel_irrel_docs(qrels[query[0]],doc_operators)
	density_op = tensor_product_T2([term], docs)
	print "density operator done"
	density_op = map(lambda l: (scipy.sparse.csc_matrix(l[0]),l[1]), density_op)
	start = time.time()
	prob_of_relevance = map(lambda l: (l[0], projections_T2(density_op,l[1])),sample_docs)
	sorted_prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	end = time.time()
	#print map(lambda l: l[0] ,sample_docs)
	print sorted_prob
	print (end-start)
	return sorted_prob

'''
target = open('many_queries_T2.txt','a')
APS =[]
APS2 = []
doc_operators = load_sparce_mats(1,1400)
print "docs_operators done"
for i in range(1,4):
	(sorted_prob1,sorted_prob2) = rel_document_sample_docs(queries[i])
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob1[:30])
	target.write(str(sorted_prob1)+'\n')
	target.write(str(AP1)+'\n')
	AP2 = calculate_AP(qrels[queries[i][0]],sorted_prob2[:30])
	target.write(str(sorted_prob2)+'\n')
	target.write(str(AP2)+'\n')
	print AP1
	print AP2
	APS += [AP1]
	APS2 += [AP2]
print calculate_MAP(APS)
print calculate_MAP(APS2)
target.flush()
target.close()
'''

doc_operators = load_sparce_mats(1,501)
toremove = []
for i in range(500):
	doc_operators[i] = (doc_operators[i][0],doc_operators[i][1].todense()) 	
	if(doc_operators[i][1].shape!=(4334,4334)):
		toremove += [i]
	print i
for index in sorted(toremove, reverse=True):
    del doc_operators[index]
print "doc_operators_done"
doc_ids = map(lambda l :l[0],doc_operators)
start = time.time()
D4s = []
#a = [13,50,11,12,55,56,100,43]
print len(doc_operators)
num = int(len(doc_operators)/4)
print num
print doc_ids
A = np.zeros((4334*4, 4334*4))
for j in range(num):
	for i in range(4):
		A[i*4334:(i+1)*4334,i*4334:(i+1)*4334] = doc_operators[i+(j*4)][1]
	D4s += [A[:]]
	print j	
end =time.time()
print (end-start)
start1 = time.time()
for j in range(num):
	D4s[j] = scipy.sparse.csc_matrix(D4s[j])
	print j
print "sparse done"
#target = ("many_queries_T2_test.txt","wb")
#target2 = ("many_queries_APS_test.txt","wb")
#target3 = ("many_queries_T1_test.txt","wb")
target4 = open("IDFs_probabilities2.txt","w")
for i in range(10,11):
	query = queries[0]
	density_ops = tensor_product_T2(query[1], docs)
	idfs = []
	Bs = []
	print len(density_ops)
	for j in range(len(density_ops)):
		B = np.zeros((4334*4, 4334*4))
		for i in range(4):
			B[i*4334:(i+1)*4334,i*4334:(i+1)*4334] = density_ops[j][0]
		B = scipy.sparse.csc_matrix(B)
		Bs += [B]
		idfs += [density_ops[j][1]]
		#idfs2 += [density_ops[2]]
	probs = [[]]*len(Bs) 
	for j in range(num):
		start = time.time()
		for i in range(len(Bs)):
			projection = Bs[i]*D4s[j]
			prob = projection.diagonal()
			probs[i] += [sum(prob[k*4334:(k+1)*4334]) for k in range(4)]
		probs[i] = zip(doc_ids,probs[i])
		end = time.time()
		print (end-start)
	probs = np.matrix(probs).transpose().tolist()
	target4.write(str(idfs)+'\n')
	target4.write(str(probs)+'\n')
end1 = time.time()
print (end-start)
target4.flush()
target4.close()












