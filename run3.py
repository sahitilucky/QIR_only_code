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


target = open("analysis1.txt","a")
target4 = open("analysis2.txt","a")
'''
ds = [486]
doc_operators = []
for i in ds:
	doc_operators += load_sparce_mats(i,i+1)
'''
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
	required_doc = filter(lambda l : l[0]== 27, doc_operators)[0]
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
docs_vectors = map(lambda l : (l[0],doc_tf_idf(l[1])), docs) 
print "doc_vectors_done"
from entropy import *

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

'''
#update document vector
i=2
doc_num = 484
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
'''

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
qs = range()
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
		(nghbr_vectors,updated_doc1) = doc_projection_update(query_subspace, required_doc, docs_vectors, q_vector,2)
		for n in nghbr_vectors:
			#print docs_vectors[n[0]-1][0]
			new_doc_vectors[n[0]-1] = n
			#analyse([1],n[1])
		new_doc_vectors[doc_num1] = (new_doc_vectors[doc_num1][0], updated_doc1)
	doc_nums = get_doc_num2(queries[i], q_vector, new_doc_vectors,doc_num1)
	print "Doc " + str(doc_nums)
	if (len(doc_nums)>0):
		doc_num = doc_nums[0]
		required_doc = filter(lambda l : l[0] == (doc_num+1), new_doc_vectors)[0][1]
		(nghbr_vectors, updated_doc) = doc_projection_update(query_subspace, required_doc, new_doc_vectors, q_vector,1)
		for n in nghbr_vectors:
			#print docs_vectors[n[0]-1][0]
			new_doc_vectors[n[0]-1] = n
			#analyse([1],n[1])
		new_doc_vectors[doc_num] = (new_doc_vectors[doc_num][0], updated_doc)
	new_doc_vectors[doc_num1] = (new_doc_vectors[doc_num1][0], updated_doc1)
	'''
	doc_nums = get_doc_num2(queries[i], q_vector, docs_vectors,doc_num1)
	print "Doc " + str(doc_nums)
	if (len(doc_nums)>0):
		doc_num = doc_nums[0]
		required_doc = filter(lambda l : l[0] == (doc_num+1), docs_vectors)[0][1]
		(nghbr_vectors, updated_doc) = doc_projection_update(query_subspace, required_doc, docs_vectors, q_vector,2)
		for n in nghbr_vectors:
			#print docs_vectors[n[0]-1][0]
			new_doc_vectors[n[0]-1] = n
			#analyse([1],n[1])
		new_doc_vectors[doc_num] = (new_doc_vectors[doc_num][0], updated_doc)
	new_doc_vectors[doc_num1] = (new_doc_vectors[doc_num1][0], updated_doc1)
	'''
	#find_near_vectors(new_doc_vectors[doc_num][1] , new_doc_vectors)
	#analyse([1],docs_vectors[doc_num][1])
	probs = map(lambda l: (l[0],cosine_similarity(q_vector.transpose(),l[1])), new_doc_vectors)
	sorted_prob = sorted(probs, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[queries[i][0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:50]	



'''
#rocchi update
from rocchio import *

for i in range(42,43):
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
	rel_docs = [docs_vectors[38]]	
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
















