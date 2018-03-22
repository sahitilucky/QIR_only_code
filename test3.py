from pysparse.sparse.pysparseMatrix import *
from pysparse.sparse import spmatrix
import time
import numpy as np
from common import *

doc_operators = load_sparce_mats(1,1400)
for i in range(200):
	doc_operators[i] = (doc_operators[i][0],doc_operators[i][1].todense()) 	

start = time.time()
a = [13,50,11,12,55,56,100,43]
A = np.zeros((4334*4, 4334*4))
for i in range(4):
  # the 10 x 10 generated matrix with "random" number
  # I'm creating it with ones for checking if the code works 
  # The random version would be:
  # b = np.random.rand(10, 10)
  # Diagonal insertion
  A[i*4334:(i+1)*4334,i*4334:(i+1)*4334] = doc_operators[a[i]][1]
A2 = np.zeros((4334*4, 4334*4))
for i in range(4):
  # the 10 x 10 generated matrix with "random" number
  # I'm creating it with ones for checking if the code works 
  # The random version would be:
  # b = np.random.rand(10, 10)
  # Diagonal insertion
  A2[i*4334:(i+1)*4334,i*4334:(i+1)*4334] = doc_operators[a[4+i]][1]
end =time.time()
print (end-start)
query = queries[0]
density_op = multi_term_mixture(query[1], docs)
B = np.zeros((4334*4, 4334*4))
for i in range(4):
  # the 10 x 10 generated matrix with "random" number
  # I'm creating it with ones for checking if the code works 
  # The random version would be:
  # b = np.random.rand(10, 10)
  # Diagonal insertion
  B[i*4334:(i+1)*4334,i*4334:(i+1)*4334] = density_op
start = time.time()
B = scipy.sparse.csc_matrix(B)
A = scipy.sparse.csc_matrix(A)
A = scipy.sparse.csc_matrix(A)
end = time.time()
start = time.time()
projection = B*A
prob = projection.diagonal()
sum1 = sum(prob[:4334])
sum2 = sum(prob[4334:4334*2])
print sum1
print sum2
end = time.time()
print (end-start)


'''
ll_mats = []
id1 = []
id2 = []
for i in range(4334):
	id1 += range(i,i+1)*4334 
	id2 += range(4334)
start = time.time()
for i in range(5):
	m = np.matrix(doc_operators[i][1])
	m = m.flatten().tolist()[0]
	L = PysparseMatrix(size = 4334)
	L.put(m, id1, id2)
	ll_mats += [L.getMatrix()]
	print
end= time.time()
print (end-start)

print sum([ll_mats[0][i,i] for i in range(4334)])
print np.trace(doc_operators[0][1])
new_ll_mat = spmatrix.matrixmultiply(ll_mats[0],ll_mats[1])
act_mat = doc_operators[0][1]*doc_operators[1][1]
print sum([new_ll_mat[i,i] for i in range(4334)])
print np.trace(act_mat)


for i in range(5):
	start = time.time()
	new_ll_mat = spmatrix.matrixmultiply(ll_mats[0],ll_mats[i])
	end = time.time()	
	print sum([new_ll_mat[j,j] for j in range(4334)])
	print "time"	
	print (end-start)
'''
