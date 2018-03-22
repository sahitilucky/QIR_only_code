import itertools
import math
from Document_rep import *
from common import allwords,IDF
from fractions import Fraction, gcd
from functools import reduce
from operator import mul
import time
#TODO : the state_vectors for term may not be good -less data
#	uniform distribution is not good
def single_term_window(t, docs):
	docs = filter(lambda l: t in l[1],docs)
	docs = map(lambda l: l[1],docs)
	weighted_set_vectors = []
	set_weighted_set_vectors = []
	window_size = 2	
	print t
	for doc in docs:
		word_array = np.array(doc)
		l = len(doc)
		ii = list(np.where(word_array == t)[0])
		for i in ii:
			t_neibr = doc[max(0,i-window_size): min(l,i+window_size+1)]
			#print t_neibr			
			state_vector = text_to_vector(t_neibr)
			weighted_set_vectors += [state_vector]
			if (state_vector not in set_weighted_set_vectors):	#to count the occurrences of a state_vector
				set_weighted_set_vectors += [state_vector]
	Nt = len(set_weighted_set_vectors)
	weighted_set_vectors = [[x,weighted_set_vectors.count(x)] for x in set_weighted_set_vectors]
	weighted_set_vectors = map(lambda l: (l[0],float(l[1])/float(Nt)), weighted_set_vectors) #uniform distribution over the weighted set 
	weighted_set_mat = map(lambda l: np.array(l[0])*math.sqrt(l[1]), weighted_set_vectors) #vectors required for density operator
	print len(weighted_set_mat)
	eigv = []
	eigw = []
	if (len(weighted_set_mat) != 0):
		(eigw, eigv) = eigenvaldecomp(weighted_set_mat,5)    #eigenvalue decomposition - to get fewer vectors
		print eigw
		density_op = density_operator(eigw, eigv, 5)
		wgt_sum =sum(eigw)
		eigw = map(lambda l : float(l)/float(wgt_sum), eigw)	
		return (density_op, zip(eigv.transpose(),eigw))
	else:
		return ([], zip(eigv,eigw))
def density_operator(eigw, eigv, maxd):
	w_sum = sum(eigw)
	eigw = map(lambda l : float(l)/float(w_sum), eigw)	
	eigw = np.array(eigw)
	eigw_diag = np.diag(eigw)
	density_op = (eigv*eigw_diag)*eigv.transpose()
	return density_op

#multi_term_mixture model
def multi_term_mixture(query, docs):
	#IDF for each term
	global IDF
	density_ops = [] 
	for t in query:
		s = single_term_window(t, docs)[0]
		if(s!=[]):
			density_ops += [IDF[t]*s]
	density_op = sum(density_ops)
	return density_op

def mixture_of_superpositions(query, docs):
	q_terms_wght_sets = []
	idfs = []
	for t in query:
		(density_op,weighted_set_vectors) = single_term_window(t, docs)
		if len(weighted_set_vectors)!=0:
			print math.sqrt(IDF[t])
			idfs += [IDF[t]]
			q_terms_wght_sets += [map(lambda l : (math.sqrt(IDF[t])*np.array(l[0]), l[1]) ,weighted_set_vectors)] 
	for t in range(len(q_terms_wght_sets)):
		q_terms_wght_sets[t] = sorted(q_terms_wght_sets[t], key = lambda l:l[1], reverse = True)[:3]
		print q_terms_wght_sets[t]
	
	weighted_set_vectors = []
	for element in itertools.product(*q_terms_wght_sets):
		l = len(element)
		print len(element)
		vector = element[0][0]
		weight = element[0][1]
		for i in range(1,l):
			vector += element[i][0]
			weight *= element[i][1]
		weighted_set_vectors += [(normalize(vector),weight)]
	weighted_set_vectors = sorted(weighted_set_vectors, key = lambda l : l[1], reverse=True)[:10]	
	wgt_sum = sum(map(lambda l : l[1],weighted_set_vectors))
	eigw = 	map(lambda l : float(l[1])/float(wgt_sum),weighted_set_vectors)
	eigv = map(lambda l : l[0].tolist()[0],weighted_set_vectors)
	print len(eigw)	
	print len(eigv)
	eigv = np.matrix(eigv).transpose()
	print eigv.shape	
	print len(weighted_set_vectors)
	if (len(weighted_set_vectors) != 0):
		density_op = density_operator(eigw, eigv, 5)	
		return ([density_op],idfs)
	else:
		return ([],idfs)

def lcm(a, b):
    return a * b // gcd(a, b)

def common_integer(numbers):
    fractions = [Fraction(n).limit_denominator() for n in numbers]
    multiple  = reduce(lcm, [f.denominator for f in fractions])
    ints      = [f * multiple for f in fractions]
    divisor   = reduce(gcd, ints)
    return [int(n / divisor) for n in ints]


#return set of density_operators corresponding to each term - exponential weight
def tensor_product_T1(query, docs):
	density_ops = []
	weights = []
	terms = []
	for t in query:
		(density_op,weighted_set_vectors) = single_term_window(t, docs)
		if(density_op != []):
			terms += [t]
			weights += [round(IDF[t],1)]		
			density_ops += [density_op]
	print terms	
	print weights
	weights = common_integer(weights)
	print weights
	while(sum(weights)>25):
		weights = list(np.array(weights)/2)
		weights = map(lambda l : round(l), weights) 
		weights = common_integer(weights)
	print weights		
	print len(density_ops)  
	density_operators = []
	for i in range(len(weights)):
		density_operators += [(density_ops[i],weights[i])]	
	print len(density_operators)	
	return density_operators

def tensor_product_T2(query, docs):
	density_ops = []
	for t in query:
		(density_op,weighted_set_vectors) = single_term_window(t, docs)
		if(density_op != []):
			density_ops += [(density_op,IDF[t])]
	idfsum = sum(map(lambda l: l[1], density_ops))
	density_ops = map(lambda l: (l[0], l[1]), density_ops)
	return density_ops

def function(weight):
	weight = (float(3)/float(2)) - (float(3)/float((weight+1)*(weight+2)))
	return weight 
#probability of relevance by squaring the projection length
def prob_projection(density_op,subspace_op):
	start = time.time()
	if(density_op.shape[1] == subspace_op.shape[0]):
		projection = density_op*subspace_op
		#projection = projection.todense()
		prob = projection.diagonal().sum()
		end = time.time()
		print end-start
		return prob
		#return np.trace(projection)
	else:
		return 0
def projections_T1(density_ops,doc):
	rels = map(lambda l: prob_projection(l[0],doc)**l[1], density_ops)
	print rels
	return reduce(mul,rels,1)

def projections_T2(density_ops,doc):
	rels = map(lambda l: (prob_projection(l[0],doc),l[1]), density_ops)
	rels1 = map(lambda l: ((1-l[1])*l[0])+(l[1]), rels)
	rels2 = map(lambda l: ((l[1])*l[0])+(1-l[1]), rels)
	print rels1[:10]
	print rels2[:10]
	return (reduce(mul,rels1,1),reduce(mul,rels2,1))

def prob_T2(idfs, probs):
	idfsum = sum(idfs)
	idfs = map(lambda l: function(float(l)/float(idfsum)), idfs)
	rels = zip(probs, idfs)
	rels1 = map(lambda l: ((1-l[1])*l[0])+(l[1]), rels)
	rels2 = map(lambda l: ((l[1])*l[0])+(1-l[1]), rels)
	#print rels1[:10]
	#print rels2[:10]
	return (reduce(mul,rels1,1),reduce(mul,rels2,1))
 
def prob_T1(weights, probs):
	weights = map(lambda l : round(l,1), weights) 
	print weights
	weights = common_integer(weights)
	print weights
	while(sum(weights)>25):
		weights = list(np.array(weights)/2)
		weights = map(lambda l : round(l), weights) 
		weights = common_integer(weights)
	print weights	
	idfs = weights
	rels = zip(probs, idfs)
	rels1 = map(lambda l: l[0]**l[1], rels)
	return (reduce(mul,rels1,1))
	'''
	rels = zip(probs, weights)
	print probs
	rels = map(lambda l : (l[1]*math.log(l[0])) if (l[0]>0) else 0 , rels)
	if 0 in rels:
		print probs
		return 0
	return (sum(rels))
	'''
def prob_mix(idfs, probs):
	idfsum = sum(idfs)
	idfs = map(lambda l: float(l)/float(idfsum), idfs)
	rels = zip(probs, idfs)
	rels1 = map(lambda l: l[0]*l[1], rels)
	return (sum(rels1))

def projection_update(densityop, subspace_op):
	subspace_op = subspace_op.todense()
	densityop = densityop.transpose().tolist()
	(eigw, eigv) = eigenvaldecomp(densityop, 5)
	cols = eigv.shape[1]
	new_eigv = np.matrix(np.zeros(eigv.shape))
	weights = []	
	for c in range(cols):
		new_eigv[:,c] = subspace_op*eigv[:,c]
		new_eigv[:,c] = normalize(new_eigv[:,c])  	# normalizing projected vector
		weights += [eigw[c]*((eigv[:,c].transpose()*subspace_op*eigv[:,c]).item())] #weight after projecting
	wgt_sum = sum(weights)
	print weights
	weights = map(lambda l: (float(l)/float(wgt_sum)), weights)
	new_op = density_operator(weights, new_eigv, 5)
	return new_op


#def doc_projection_update(density_op, subspace_op):
#	= eigenvaldecomp(subspace_op, )
	



