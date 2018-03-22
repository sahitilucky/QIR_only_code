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

#Parameter: 2 time pseudo relevance feedback, 2 rel_docs for updation, total 10 docs taken to check

def rel_update(query, term_probs_pt, term_probs_ut, term_probs_ct, rel_docs, non_rel_docs):
	rel_docs = map(lambda l : docs[l[0]-1], rel_docs)
	non_rel_docs = map(lambda l : docs[l[0]-1], non_rel_docs) 
	k = len(rel_docs)	
	if(k == 0):
		return (term_probs_pt, term_probs_ut, term_probs_ct)	
	for q in query:
		vrt = 0	
		for r in rel_docs:
			if q in r[1]:
				vrt += 1
		vr = len(rel_docs)
		term_probs_pt[q] = float(vrt + k*term_probs_pt[q])/ float(vr + k) 
	for q in query:
		nvrt = 0	
		for r in non_rel_docs:
			if q in r[1]:
				nvrt += 1
		nvr = len(non_rel_docs)
		term_probs_ut[q] = float(nvrt + k*term_probs_ut[q])/ float(nvr + k) 
	for q in query:
		term_probs_ct[q] = math.log(float(term_probs_pt[q])/float(1-term_probs_pt[q])) + math.log(float(1-term_probs_ut[q])/float(term_probs_ut[q])) 

	return (term_probs_pt, term_probs_ut, term_probs_ct)

	

def psuedo_rel_update(query, term_probs_pt, term_probs_ut, term_probs_ct, rel_docs, non_rel_docs):
	rel_docs = map(lambda l : docs[l[0]-1], rel_docs)
	non_rel_docs = map(lambda l : docs[l[0]-1], non_rel_docs) 
	for q in query:
		vrt = 0	
		for r in rel_docs:
			if q in r[1]:
				vrt += 1
		vr = len(rel_docs)
		term_probs_pt[q] = float(vrt + 0.5*term_probs_pt[q])/ float(vr + 1) 
	for q in query:
		nvrt = 0	
		for r in non_rel_docs:
			if q in r[1]:
				nvrt += 1
		nvr = len(non_rel_docs)
		term_probs_ut[q] = float(nvrt + 0.5*term_probs_ut[q])/ float(nvr + 1) 
	for q in query:
		term_probs_ct[q] = math.log(float(term_probs_pt[q])/float(1-term_probs_pt[q])) + math.log(float(1-term_probs_ut[q])/float(term_probs_ut[q])) 

	return (term_probs_pt, term_probs_ut, term_probs_ct)

	

for q_num in range(50):
	query = queries[q_num]
	term_probs_pt = {}
	term_probs_ut = {}
	term_probs_ct = {}
	print "Query " + str(q_num+1)
	for q in query[1]:
		term_probs_pt[q] = 0.5
		term_probs_ut[q] = float(1)/float(math.exp(idf[q]) - 1) 
		term_probs_ct[q] = math.log(float(term_probs_pt[q])/float(1-term_probs_pt[q])) + math.log(float(1-term_probs_ut[q])/float(term_probs_ut[q])) 
	probs = []
	for d in docs:
		score = 0
		for q in query[1]:
			if q in d[1]:
				score += term_probs_ct[q]
		probs += [(d[0], score)]
	sorted_prob = sorted(probs, key = lambda l:l[1], reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:30]
	#pseudo relevance feedback
	for i in range(2):
		rel_docs = sorted_prob[:10]
		#rel_doc_nums = map(lambda l : l[0], sorted_prob)
		non_rel_docs = sorted_prob[10:]
		(term_probs_pt, term_probs_ut, term_probs_ct) = psuedo_rel_update(query[1], term_probs_pt, term_probs_ut, term_probs_ct, rel_docs, non_rel_docs)
	
		probs = []
		for d in docs:
			score = 0
			for q in query[1]:
				if q in d[1]:
					score += term_probs_ct[q]
			probs += [(d[0], score)]
			
		sorted_prob = sorted(probs, key = lambda l:l[1], reverse = True)
		AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:30]

	#relevance feedback
	sorted_prob=sorted_prob[:30] 
	qs = map(lambda l : l[0], qrels[query[0]]) 
	rel_docs = []
	non_rel_docs = []	
	for s in sorted_prob:
		if s[0] in qs:
			doc_num = s[0]-1
			rel_docs += [s]
		else:
			non_rel_docs += [s]
	rel_docs = rel_docs[:2]	
	non_rel_docs = non_rel_docs[:]
	print "Docs " + str(rel_docs)
	(term_probs_pt, term_probs_ut, term_probs_ct) = rel_update(query[1], term_probs_pt, term_probs_ut, term_probs_ct, rel_docs, non_rel_docs)

	probs = []
	for d in docs:
		score = 0
		for q in query[1]:
			if q in d[1]:
				score += term_probs_ct[q]
		probs += [(d[0], score)]
		
	sorted_prob = sorted(probs, key = lambda l:l[1], reverse = True)
	AP1 = calculate_AP(qrels[query[0]],sorted_prob[:30])
	print AP1
	print sorted_prob[:30]


