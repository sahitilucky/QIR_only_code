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


target = open("analysis11.txt","a")
target4 = open("analysis21.txt","a")
'''
ds = [486]
doc_operators = []
for i in ds:
	doc_operators += load_sparce_mats(i,i+1)
'''
doc_operators = load_sparce_mats(1,1401)
probs1 = []
probs2 =[]
probs3 =[]
for i in range(10,11):
	query = queries[i]
	target.write('Query '+ str(query[0])+'\n')
	target4.write('Query '+str(query[0])+'\n')
	density_ops = tensor_product_T2(query[1], docs)
	idfs = map(lambda l : l[1],density_ops)
	#updating density_ops using a document
	required_doc = filter(lambda l : l[0]== 28, doc_operators)[0]
	density_ops = map(lambda l: (projection_update(l[0],required_doc[1]),l[1]), density_ops)
	#updating density_ops using a document
	#checkdocs = [495,20,27,28,262,263,160,654,556,1327,1356]
	#required_docs = filter(lambda l : l[0] in checkdocs, doc_operators)
	density_ops = map(lambda l: (scipy.sparse.csc_matrix(l[0]),l[1]), density_ops)
	probs = map(lambda l : (l[0], idf_probabilities(density_ops,l[1],l[0])), doc_operators)
	target4.write(str(idfs)+'\n')
	target4.write(str(probs)+'\n')
	d = doc_operators[0][1]
	probs1 = map(lambda l : (l[0], prob_mix(idfs,l[1])),probs)
	probs2 = map(lambda l: (l[0], prob_T2(idfs,l[1])),probs)
	probs3 = map(lambda l :(l[0], prob_T1(idfs,l[1])),probs)
	sorted_prob1 = sorted(probs1, key = lambda l:l[1],reverse = True)
	sorted_prob2 = sorted(probs2, key = lambda l:l[1][0],reverse = True)
	sorted_prob3 = sorted(probs2, key = lambda l:l[1][1],reverse = True)
	sorted_prob4 = sorted(probs3, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob1[:30])
	print AP1
	print sorted_prob1[:50]	
	target.write(str(sorted_prob1[:50])+'\n')
	target.write(str(AP1)+'\n')
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob2[:30])
	print AP1
	print sorted_prob2[:50]	
	target.write(str(sorted_prob2[:50])+'\n')
	target.write(str(AP1)+'\n')
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob3[:30])
	print AP1
	print sorted_prob3[:50]	
	target.write(str(sorted_prob3[:50])+'\n')
	target.write(str(AP1)+'\n')
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob4[:30])
	print AP1
	print sorted_prob4[:50]	
	target.write(str(sorted_prob4[:50])+'\n')
	target.write(str(AP1)+'\n')
target.flush()	
target.close()
target4.flush()
target4.close()

'''
flight,aircraft
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
prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
print prob
'''


'''
APs = []
for i in range(10):
	#getting relevant docs using mixture method
	density_op = multi_term_mixture(queries[i][1], docs)
	print "density operator done"
	prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),doc_operators)
	prob = sorted(prob_of_relevance, key = lambda l:l[1],reverse = True)
	print prob[:10]
	print queries[i][0]
	AP = calculate_AP(qrels[queries[i][0]],prob[:10])
	APs += [AP]
	print AP
print calculate_MAP(APs)

#getting relevant docs using mixture of superpositions method
density_op,weight_set_vectors = mixture_of_superpositions(queries[0][1], docs)
prob_of_relevance = map(lambda l: (l[0], prob_projection(density_op,l[1])),doc_operators)
prob = sorted(prob_of_relevance,key = lambda l:l[1], reverse = True)
print prob[:10]

APs = []
for i in range(10):
	#getting relevant docs using T2 method
	density_ops = tensor_product_T2(queries[i][1], docs)
	prob_of_relevance = map(lambda l: (l[0], projections_T2(density_ops,l[1])),doc_operators)
	prob = sorted(prob_of_relevance,key = lambda l:l[1],reverse = True)
	print prob[:10]
	AP = calculate_AP(qrels[queries[i][0]],prob[:10])
	APs += [AP]
	print AP
print calculate_MAP(APs)


#getting relevant docs using T1 method
density_ops = tensor_product_T1(queries[0][1], docs)
prob_of_relevance = map(lambda l: (l[0], projections_T1(density_ops,l[1])),doc_operators[:30])
prob = sorted(prob_of_relevance,key = lambda l:l[1],reverse = True)
print prob[:10]
calculate_AP(qrels[queries[0][0]],prob[:10])
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
prob = sorted(prob_of_relevance,key = lambda l:l[1],reverse = True)
print prob[:10]
'''












