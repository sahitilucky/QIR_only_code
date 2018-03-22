from common import *
import pickle
import scipyss
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





