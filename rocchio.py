def rocchio(query, rel_docs, non_rel_docs):
	alpha = 1
	beta = 0.75
	gamma = 0
	rel_docs = map(lambda l: l[1], rel_docs)
	non_rel_docs = map(lambda l: l[1], non_rel_docs)	
	rel_l = len(rel_docs)	
	rel_c = sum(rel_docs)/float(rel_l)
	non_rel_l = len(non_rel_docs)	
	non_rel_c = sum(non_rel_docs)/float(non_rel_l)
	query = alpha*query + ((beta*rel_c) - (gamma*non_rel_c))
	return query

def split_rel_non_rel_docs(query, sorted_prob,qrels):
	qrels = qrels[query[0]]
	qs = map(lambda l : l[0], qrels) 
	rel_docs = []
	non_rel_docs = []	
	for s in sorted_prob:
		if s[0] in qs:
			doc_num = s[0]-1
			rel_docs += [s]
		else:
			non_rel_docs += [s]
	rel_docs = rel_docs[:1]	
	non_rel_docs = non_rel_docs[:1]
	return (rel_docs, non_rel_docs)

	
