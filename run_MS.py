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

def idf_probabilities(density_ops,d,num):
	probs = []
	for i in range(len(density_ops)):
		B = density_ops[i]	
		probs += [prob_projection(B,d)]
	print ("over "+str(num))
	return (probs)



target = open("MS_many_queries.txt","a")
target4 = open("MS_IDFs_probabilities.txt","a")
doc_operators = load_sparce_mats(1,1401)
'''
toremove = []
for i in range(1400):
	doc_operators[i] = (doc_operators[i][0],doc_operators[i][1].todense()) 	
	if(doc_operators[i][1].shape!=(4334,4334)):
		toremove += [i]
	print i
for index in sorted(toremove, reverse=True):
    del doc_operators[index]
print "doc_operators_done"
print len(toremove)
print len(doc_operators)
'''
ids = [29,31,33,35]
for i in ids:
	query = queries[i]
	(density_ops,idfs) = mixture_of_superpositions(query[1],docs)
	density_ops = map(lambda l: scipy.sparse.csc_matrix(l), density_ops)
	target.write('Query '+ str(query[0])+'\n')
	target4.write('Query '+str(query[0])+'\n')
	probs1 = map(lambda l : (l[0], idf_probabilities(density_ops,l[1],l[0])), doc_operators)
	target4.write(str(idfs)+'\n')
	target4.write(str(probs1)+'\n')
	sorted_prob1 = sorted(probs1, key = lambda l:l[1],reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob1[:30])
	print AP1
	print sorted_prob1[:50]	
	target.write(str(sorted_prob1[:50])+'\n')
	target.write(str(AP1)+'\n')

target.flush()	
target.close()
target4.flush()
target4.close()


