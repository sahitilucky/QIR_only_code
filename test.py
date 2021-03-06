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



s = [(498, 6.726454144308931e-19), (266, 2.9760846262911905e-19), (1255, 1.5813407209353488e-19), (927, 1.2235141515631762e-19), (1006, 1.0211433334730428e-19), (494, 6.878671112727585e-20), (447, 6.465035472519446e-20), (248, 4.1021023096987984e-20), (435, 3.0563123280608485e-20), (94, 2.5018637503846204e-20), (1328, 1.5947655212553485e-20), (297, 1.0041762431165526e-20), (468, 9.789992824865667e-21), (1352, 9.367728165534088e-21), (85, 8.353552061876843e-21), (197, 7.31364428856917e-21), (93, 5.946502680097692e-21), (211, 5.1626882941672305e-21), (814, 3.7116714875813085e-21), (1274, 2.6114351064029257e-21), (1238, 2.423398763846913e-21), (433, 2.328943667909803e-21), (1339, 2.0287598944586795e-21), (556, 1.964661618177585e-21), (1319, 1.8443855396101694e-21), (193, 1.832919447663667e-21), (370, 1.695948666187111e-21), (1310, 1.694685779983764e-21), (918, 1.6807131861234357e-21), (662, 1.485966432142095e-21), (465, 1.2861220995358094e-21), (1222, 1.2437907719456752e-21), (493, 1.1831784479932507e-21), (626, 1.1764590662912807e-21), (259, 1.009364934026573e-21), (160, 9.89362190865296e-22), (231, 8.361928854333938e-22), (443, 5.738600971082077e-22), (1393, 4.983875554782699e-22), (666, 4.644378852507054e-22), (943, 4.605344915609331e-22), (1253, 4.434796343665343e-22), (994, 4.139468596535101e-22), (415, 3.717232853826185e-22), (1074, 3.586028693951108e-22), (19, 3.5569206358827025e-22), (124, 3.408694143713485e-22), (372, 3.386264237320671e-22), (924, 3.386264237320671e-22) ]


AP1 = calculate_AP(qrels[queries[15][0]],s[:30])
print AP1

