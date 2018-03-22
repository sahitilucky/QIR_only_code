from common import *
import itertools
import numpy as np
from numpy import linalg as LA
#from sympy import *



#aspect vector created using binary weighted scheme
def text_to_vector(fragment):
	global allwords
	as_vector = map(lambda l: 1 if l in fragment else 0, allwords)
	#print fragment	
	return list(normalize(as_vector))

#assuming tokens of document as input
#forming document subspace in termspace - should consider term relations in future may be by LSA
def divide_fragments(d):
	window_size = 5
	doc_length = len(d)
	numfragments = doc_length/window_size
	if( doc_length % window_size != 0):
		numfragments += 1
	all_fragments = []
	for frag_num in range(numfragments):
		fragment = d[frag_num*window_size : frag_num*window_size + window_size]
		fragment = text_to_vector(fragment)
		all_fragments += [fragment]
	(eigw, eigv) = eigenvaldecomp(all_fragments,10)
	subspace_operator = subspace_op(eigv, 10)
	return subspace_operator
	
#constraining number of dimension to 25 at max.	
#assuming equal weight for all fragments as they did not mention about weights
#check for orthogonality and symmetric matrices
def eigenvaldecomp(asp_vectors,maxd):
	l = len(asp_vectors)
	s_mat = np.matrix(asp_vectors)
	s_mat = s_mat.transpose()*s_mat   
	eigw, eigv = LA.eigh(s_mat)
	cols = eigv.shape[1]	
	eigw = eigw[-min(maxd,cols,l):]
	eigv = eigv[:, -min(maxd,cols,l):]
	return (eigw, eigv)


def subspace_op(eigv, maxd):
	'''	
	s_mat = (np.matmul(eigv[:,[cols-1]], eigv[:,[cols-1]].transpose()))
	dim = 1	
	for i in range(1,min(cols,cols)):
		#if(eigw[cols-i-1] < 0):
		#	break
		dim+=1		
		s_mat += np.matmul(eigv[:,[cols-i-1]], eigv[:,[cols-i-1]].transpose())	
	print dim
	'''
	#TODO round off very small eigw to 0 
	#why fix top k vectors - may be x% variance will be good 
	#cols = eigv.shape[1]	
	#eigv = eigv[:,-min(maxd,cols,l):] 
	print eigv.shape
	sub_operator = eigv*eigv.transpose()	
	return sub_operator
	
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def text_to_vector2(fragment):
	global allwords
	as_vector = map(lambda l: 1 if l in fragment else 0, allwords)
	#print fragment	
	return as_vector
	
def doc_tf_idf(d):
	window_size = 5s
	doc_length = len(d)
	numfragments = doc_length/window_size
	if( doc_length % window_size != 0):
		numfragments += 1
	all_fragments = []
	for frag_num in range(numfragments):
		fragment = d[frag_num*window_size : frag_num*window_size + window_size]
		fragment = text_to_vector2(fragment)
		all_fragments += [fragment]
	all_fragments = map(lambda l: np.array(l),all_fragments)
	#sum all fragments
	if(all_fragments ==[]):
		return np.matrix(np.zeros([4334,1]))
	all_fragments = sum(all_fragments) #without normalisation
	tf_idf = zip(allwords,all_fragments.tolist())
	tf_idf = map(lambda l : (IDF[l[0]]*l[1]), tf_idf)	 
	tf_idf = normalize(tf_idf)			#with normalisation
	return np.matrix(tf_idf).transpose()
	
def get_query_subspace(density_ops):
	wgt_sum = sum(map(lambda l : l[1], density_ops))
	density_op = sum(map(lambda l : l[0]*(float(l[1])/float(wgt_sum)) , density_ops))
	(eigw, eigv) = eigenvaldecomp(density_op, len(density_ops)*3)
	query_subspace = subspace_op(eigv,5)
	analyse(eigw,eigv)	
	return query_subspace

def analyse(eigw,eigv):
	c = eigv.shape[1]
	print (eigv[:,0].transpose()*eigv[:,1])
	print (eigv[:,1].transpose()*eigv[:,2])	
	for i in range(c):
		print eigw[i]
		rels = []
		for w in range(len(allwords)):
			rels += [(allwords[w],eigv[w,i])]
		prob = sorted(rels, key = lambda l:l[1],reverse = True)
		print prob[:100]

def get_threshold(similarities1):
	std = np.std(similarities1)
	avg = float(sum(similarities1))/float(len(similarities1))	
	histogram = {}
	for s in similarities1:
		s = round(s,2)
		if s not in histogram:
			histogram[s] = 1
		else:
			histogram[s] += 1
	histogram = histogram.items()
	histogram = sorted(histogram, key = lambda l :l[0], reverse = True)
	#print histogram	
	for h in range(len(histogram)):
		if histogram[h][1] > 30:
			trshd = histogram[h-1][0]  
			break
	return trshd

def find_near_vectors_threshold(doc_tf_idf,alldocs):
	nghbr_vectors = map(lambda l : (l[0], cosine_similarity(doc_tf_idf.transpose(), l[1])), alldocs)
	nghbr_vectors = sorted(nghbr_vectors, key= lambda l : l[1],reverse= True)
	similarities = 	map(lambda l: l[1], nghbr_vectors)	
	#print similarities
	avg = float(sum(similarities))/float(len(similarities))	
	#print avg
	threshold = get_threshold(similarities)	
	#print threshold
	nghbr_vectors = filter(lambda l : l[1] > threshold, nghbr_vectors)
	nghbr_vectors = map(lambda l : l[0], nghbr_vectors)
	all_docs = filter(lambda l: l[0] in nghbr_vectors,alldocs)
	#print len(all_docs)	
	return all_docs

def find_near_vectors(doc_tf_idf,alldocs):
	nghbr_vectors = map(lambda l : (l[0], cosine_similarity(doc_tf_idf.transpose(), l[1])), alldocs)
	nghbr_vectors = sorted(nghbr_vectors, key= lambda l : l[1],reverse= True)[:40]
	#print nghbr_vectors
	all_docs = alldocs[:]
	nghbr_vectors = map(lambda l : all_docs[l[0]-1], nghbr_vectors)
	#all_docs = filter(lambda l: l[0] in nghbr_vectors,alldocs)
	return nghbr_vectors


def query_doc_projection_update(required_doc, alldocs, q_vector,alpha,query_terms):
	#print "query_terms:"
	#print len(query_terms)
	nghbrs_vectors = find_near_vectors(required_doc,alldocs)
	query_subspace = q_vector*q_vector.transpose()
	doc_subspace = required_doc*required_doc.transpose()
	updated_query = doc_subspace*q_vector 
	#if(len(query_terms)<=7):
	updated_query = normalize((q_vector + (0.75*required_doc))/float(2))
	#else:
	#updated_query = normalize((q_vector + (required_doc))/float(2))
	new_query_subspace = updated_query*updated_query.transpose()	
	updated_doc = new_query_subspace*required_doc	
	updated_doc = normalize(updated_doc )
	
	for i in range(20):
		updated = new_query_subspace*nghbrs_vectors[i][1]
		#print nghbrs_vectors[i][0]
		sim =  cosine_similarity(required_doc.transpose(), nghbrs_vectors[i][1])
		#print cosine_similarity(updated.transpose(), updated_query)
		#print sim		
		#print cosine_similarity(nghbrs_vectors[i][1].transpose(), updated_query)		
		updated = (updated*sim*alpha + nghbrs_vectors[i][1])
		updated = normalize(updated)	
		#print cosine_similarity(updated.transpose(), updated_query)
		nghbrs_vectors[i] = (nghbrs_vectors[i][0], updated)
	for i in range(20, 40):
		updated = new_query_subspace*nghbrs_vectors[i][1]
		#print nghbrs_vectors[i][0]
		sim =  cosine_similarity(required_doc.transpose(), nghbrs_vectors[i][1])
		#print cosine_similarity(updated.transpose(), updated_query)
		#print sim		
		#print cosine_similarity(nghbrs_vectors[i][1].transpose(), updated_query)		
		updated = ((updated*sim*0.5*alpha) + nghbrs_vectors[i][1])
		updated = normalize(updated)	
		#print cosine_similarity(updated.transpose(), updated_query)
		nghbrs_vectors[i] = (nghbrs_vectors[i][0], updated)
	return  (nghbrs_vectors, updated_doc, updated_query)
		
def query_doc_projection_update_rnr(required_doc, alldocs, q_vector):
	nghbrs_vectors = find_near_vectors_threshold(required_doc,alldocs)
	query_subspace = q_vector*q_vector.transpose()
	updated_doc = query_subspace*required_doc
	doc_subspace = required_doc*required_doc.transpose()
	updated_query = doc_subspace*q_vector 
	updated_doc = normalize((required_doc + updated_doc)/float(2))
	updated_query = normalize((q_vector + updated_query)/float(2))
	new_query_subspace = updated_query*updated_query.transpose()	
	for i in range(len(nghbrs_vectors)):
		updated = new_query_subspace*nghbrs_vectors[i][1]
		#print nghbrs_vectors[i][0]
		sim =  cosine_similarity(required_doc.transpose(), nghbrs_vectors[i][1])
		#print cosine_similarity(updated.transpose(), updated_query)
		#print sim		
		#print cosine_similarity(nghbrs_vectors[i][1].transpose(), updated_query)		
		updated = (updated*sim + nghbrs_vectors[i][1])
		updated = normalize(updated)	
		#print cosine_similarity(updated.transpose(), q_vector)
		nghbrs_vectors[i] = (nghbrs_vectors[i][0], updated)
	return  (nghbrs_vectors, updated_doc, updated_query)

		
def doc_projection_update(query_subspace,doc_tf_idf,alldocs,q_vector, alpha):
	nghbrs_vectors = find_near_vectors_threshold(doc_tf_idf,alldocs)
	updated_doc = query_subspace*doc_tf_idf
	updated_doc = normalize(updated_doc)
	change_v = updated_doc - doc_tf_idf
	doc_subspace = doc_tf_idf*doc_tf_idf.transpose()
	
	for i in range(len(nghbrs_vectors)):
		#analyse([1], nghbrs_vectors[i][1])
		updated = query_subspace*nghbrs_vectors[i][1]
		proj_doc = doc_subspace*nghbrs_vectors[i][1]
		#print nghbrs_vectors[i][0]
		updated = updated - nghbrs_vectors[i][1]
		#sim = 0.5		
		sim =  cosine_similarity(doc_tf_idf.transpose(), nghbrs_vectors[i][1])*alpha
		if sim>1:
			sim = 0.99		
		#print cosine_similarity(proj_doc.transpose(), q_vector)
		#print cosine_similarity(nghbrs_vectors[i][1].transpose(), q_vector)		
		#updated = (updated + proj_doc + nghbrs_vectors[i][1])/float(3)
		#updated = normalize(updated)	
		#print cosine_similarity(updated.transpose(), q_vector)
		#print sim		
		#print cosine_similarity(nghbrs_vectors[i][1].transpose(), q_vector)		
		updated = (updated*sim + nghbrs_vectors[i][1])
		updated = normalize(updated)	
		#print cosine_similarity(updated.transpose(), q_vector)
		nghbrs_vectors[i] = (nghbrs_vectors[i][0], updated)
		
	return (nghbrs_vectors,updated_doc)

def cosine_similarity(v1,v2):
	if(np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0):
		return 0
	return float(np.dot(v1,v2))/float(np.linalg.norm(v1)*np.linalg.norm(v2))
