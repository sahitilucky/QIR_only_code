'''
f = open("many_queries_all_test.txt" , 'r')
mix_aps = []
t2_aps= []
t2_ap2s = []
t1_aps = []
line = f.readline()
query = line.strip()
query_stats = {}
while line:
	line = f.readline()
	line = f.readline()
	mix_ap = float(line.strip())
	print mix_ap
	query_stats[query] = [("mix_ap", mix_ap)]
	mix_aps += [mix_ap] 
	line = f.readline()
	line = f.readline()
	t2_ap = float(line.strip())
	print t2_ap
	query_stats[query] += [("t2_ap", t2_ap)]
	t2_aps += [t2_ap] 
	line = f.readline()
	line = f.readline()
	t2_ap2 = float(line.strip())
	print t2_ap2
	query_stats[query] += [("t2m_ap", t2_ap2)]
	t2_ap2s += [t2_ap2] 
	line = f.readline()
	line = f.readline()
	t1_ap = float(line.strip())
	print t1_ap
	query_stats[query] += [("t1_ap", t1_ap)]
	t1_aps += [t1_ap]
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print len(mix_aps)
f = open("many_queries_all_test3.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	line = f.readline()
	mix_ap = float(line.strip())
	print mix_ap
	query_stats[query] = [("mix_ap", mix_ap)]
	mix_aps += [mix_ap] 
	line = f.readline()
	line = f.readline()
	t2_ap = float(line.strip())
	print t2_ap
	query_stats[query] += [("t2_ap", t2_ap)]
	t2_aps += [t2_ap] 
	line = f.readline()
	line = f.readline()
	t2_ap2 = float(line.strip())
	print t2_ap2
	query_stats[query] += [("t2m_ap", t2_ap2)]
	t2_ap2s += [t2_ap2] 
	line = f.readline()
	line = f.readline()
	t1_ap = float(line.strip())
	print t1_ap
	query_stats[query] += [("t1_ap", t1_ap)]
	t1_aps += [t1_ap]
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print len(mix_aps)

f = open("many_queries_all_test4.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	line = f.readline()
	mix_ap = float(line.strip())
	print mix_ap
	query_stats[query] = [("mix_ap", mix_ap)]
	mix_aps += [mix_ap] 
	line = f.readline()
	line = f.readline()
	t2_ap = float(line.strip())
	print t2_ap
	query_stats[query] += [("t2_ap", t2_ap)]
	t2_aps += [t2_ap] 
	line = f.readline()
	line = f.readline()
	t2_ap2 = float(line.strip())
	print t2_ap2
	query_stats[query] += [("t2m_ap", t2_ap2)]
	t2_ap2s += [t2_ap2] 
	line = f.readline()
	line = f.readline()
	t1_ap = float(line.strip())
	print t1_ap
	query_stats[query] += [("t1_ap", t1_ap)]
	t1_aps += [t1_ap]
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print len(mix_aps)


print "MAPS"
print len(mix_aps)
print float(sum(mix_aps))/float(len(mix_aps))
print float(sum(t2_aps))/float(len(t2_aps))
print float(sum(t2_ap2s))/float(len(t2_ap2s))
print float(sum(t1_aps))/float(len(t1_aps))


tf_aps = []
f = open("tf_idf_results.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	line = f.readline()
	tf_ap = float(line.strip())
	if query in query_stats.keys():
		query_stats[query] += [("tf_ap", tf_ap)]
		#else:
		#	query_stats[query] = [("tf_ap", tf_ap)]
		tf_aps += [tf_ap] 
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print len(tf_aps)
print float(sum(tf_aps))/float(len(tf_aps))

qs = query_stats.keys()
fs = [0,0,0,0,0]
for q in qs:
	sq = sorted(query_stats[q], key = lambda l : l[1], reverse = True)
	print q
	print sq
	if(sq[0][0] == 'mix_ap'):
		fs[0] += 1
	if(sq[0][0] == 't2_ap'):
		fs[1] += 1
	if(sq[0][0] == 't2m_ap'):
		fs[2] += 1
	if(sq[0][0] == 't1_ap'):
		fs[3] += 1
	if(sq[0][0] == 'tf_ap'):
		fs[4] += 1
print fs



ms_aps = []
f = open("MS_many_queries.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	line = f.readline()
	ms_ap = float(line.strip())
	ms_aps += [ms_ap] 
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print len(ms_aps)
print float(sum(ms_aps))/float(len(ms_aps))
'''
print "----------------------Doc updation testing--------------------------------"
from common import *
import pickle
import scipy
import time
#docs = read_data_docs('../../cran/cran.all.1400', 1400)
#print docs[:3]
#pickle.dump(docs, open("../../cran/cran_1400_2.pkl", "wb"))
#docs = pickle.load(open("../../cran/cran_1400_2.pkl" ,"rb"))
queries = read_queries('../../cran/cran.qry')
qrels = read_query_rels('../../cran/cranqrel')
print "--------Queries-------"

after_aps1 = []
before_aps1 = []
neg_eg_qs = []
f = open("rocchio_update.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	before_ap = float(line.strip()) 
	line = f.readline()
	line = f.readline()
	line = f.readline()
	
	after_ap = float(line.strip())
	line = f.readline()
	after_aps1 += [after_ap]
	before_aps1 += [before_ap]
	if(after_ap < before_ap):
		neg_eg_qs += [int(query.split("Query")[1])]
		print query
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print len(after_aps1)
print "BEFORE AP:"
print float(sum(before_aps1))/float(len(before_aps1))
print "AFTER AP:"
print float(sum(after_aps1))/float(len(after_aps1))



after_aps = []
before_aps = []
neg_eg_docs = []
f = open("doc_updation_20.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	before_ap = float(line.strip()) 
	line = f.readline()
	line = f.readline()
	line = f.readline()
	
	#print line
	after_ap = float(line.strip())
	line = f.readline()
	after_aps += [after_ap]
	before_aps += [before_ap]
	if(after_ap < before_ap):
		neg_eg_docs += [int(query.split("Query")[1])]
		print query 
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print after_aps
print len(after_aps)
print "BEFORE AP:"
print float(sum(before_aps))/float(len(before_aps))
print "AFTER AP:"
print float(sum(after_aps))/float(len(after_aps))


pos_cases = []
neg_cases = []
for i in range(len(after_aps)):
	if(after_aps1[i] < after_aps[i]):
		pos_cases += [i+1]
	if(after_aps1[i] > after_aps[i]):
		neg_cases += [i+1]
print "Doc updation better:"
print pos_cases
pos_cases = map(lambda l : (l, len(queries[l-1][1])),pos_cases)
print pos_cases
print "Query updation better:"
print neg_cases
neg_cases = map(lambda l : (l, len(queries[l-1][1])),neg_cases)
print neg_cases

'''
print "--------------------Doc updation better-----------------------------"
for i in pos_cases:
	print "Query " + str(i+1)
	print queries[i]
	print qrels[queries[i][0]]

print "--------------------Query updation better-------------------------"
for i in neg_cases:
	print "Query " + str(i+1)
	print queries[i]python
	print qrels[queries[i][0]]

print "--------------------Doc negative examples-----------------------------"
for i in neg_eg_docs:
	print "Query " + str(i)
	print queries[i-1]
	print qrels[queries[i-1][0]]

print "--------------------Query negative examples-------------------------"
for i in neg_eg_qs:
	print "Query " + str(i)
	print queries[i-1]
	print qrels[queries[i-1][0]]
'''


'''
print "------------------Probabilistic model-------------------------"
after_aps = []
before_aps = []
neg_eg = 0
f = open("probabilistic_output.txt" , 'r')
line = f.readline()
query = line.strip()
while line:
	line = f.readline()
	line = f.readline()
	line = f.readline()
	before_ap = float(line.strip()) 
	line = f.readline()
	line = f.readline()
	line = f.readline()
	after_ap = float(line.strip())
	line = f.readline()
	after_aps += [after_ap]
	before_aps += [before_ap]
	if(after_ap < before_ap):
		neg_eg += 1
		print query 
	line=f.readline() 
	query = line.strip()
f.close()
print "len"
print after_aps
print len(after_aps)
print "BEFORE AP:"
print float(sum(before_aps))/float(len(before_aps))
print "AFTER AP:"
print float(sum(after_aps))/float(len(after_aps))
print neg_eg
'''







