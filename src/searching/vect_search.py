from src.searching.index_reader import FreqIndex
import src.searching.weightings as w
from src.language_processing.processing import Collection
from config import QUERIES_DIR
from collections import Counter
import os


class WrongFormatError(Exception): pass


# Vectorial queries will be represented as a text (string)
def read_queries(query_file='query.text'):
    """ Read queries that are written on query_file """
    queries = list()

    query_path = os.path.join(QUERIES_DIR, query_file)
    with open(query_path, 'r') as f:
        take_line = False
        query = ""

        for line in f:
            if take_line:
                if line[:2] not in ['.I', '.W', '.A', '.N']:
                    query += line
                else:
                    take_line = False
                    queries.append(query)

            if line.startswith('.W'):
                take_line = True
                query = ""

    return queries


def start_search_engine_cli(collection_name, index=None, tf=w.tf, idf=w.idf, rsv=w.rsv_cos):
    """ Input queries in Command Line Interface and search for results immediately after each request """
    print("Entrez vos requêtes en ligne de commande sous forme textuelle (comme dans un moteur de recherche standard)")
    print('______________________\n')

    running = True
    if not index:
        index = FreqIndex(collection_name)
    while running:
        query = input_query()
        query_tokens = Collection(None).process(query)
        display_query(query_tokens)
        result = search_for_query(query_tokens, index, tf, idf, rsv)
        display_result(result)

        print("Voulez vous saisir une nouvelle requête ? (Y/N)")
        answer = input().lower()
        while not answer in ['y', 'n']:
            print("Entrée invalide. Voulez vous saisir une nouvelle requête ? (Y/N)")
            answer = input().lower()
        if answer == 'n':
            running = False
    print("")


def input_query():
    # Write query in command line
    print("\nNouvelle requête :")
    return input()


def display_query(query_tokens):

    print("\nRequête après traitement :", ' '.join(query_tokens))
    print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n')


def search_for_query(query_tokens, index, tf=w.tf_log, idf=w.idf, rsv=w.rsv_cos):

    query_index = Counter()
    for token in query_tokens:
        query_index[token] += 1

    total_docs = len(index.get_all_documents())
    scores = [0] * total_docs
    normalize_q = []

    terms_index = index.find_documents(query_index.keys())
    relevant_docs = set()
    for term, docs in terms_index.items():
        relevant_docs.update(docs[1].keys())
    relevant_docs = sorted(list(relevant_docs))
    terms_by_doc = index.get_related_terms(relevant_docs)
    all_doc_freqs = index.get_all_doc_freqs()

    for term in terms_index:
        doc_freq, postings = terms_index[term]
        wq = tf(query_index[term], query_index.values()) * idf(doc_freq, total_docs)
        normalize_q.append(wq)

        for doc, freq in postings.items():
            wd = tf(freq, terms_by_doc[doc].values()) * idf(doc_freq, total_docs)
            scores[doc] += wd * wq

    for doc_id in relevant_docs:
        terms_dict = terms_by_doc[doc_id]
        normalize_d = []
        for term_id, freq_term in terms_dict.items():
            wtd = tf(freq_term, terms_by_doc[doc_id].values()) * idf(all_doc_freqs[term_id], total_docs)
            normalize_d.append(wtd)
        scores[doc_id] = rsv(scores[doc_id], normalize_q, normalize_d)

    sorted_docs = sorted(relevant_docs, key=lambda d: -scores[d])

    return index.get_documents_from_ids(sorted_docs[:100])


def display_result(list):
    print("Liste des résultats :")
    if len(list) == 0:
        print("Aucun résultat")
    for rank, doc_title in enumerate(list):
        print("%i) %s" % (rank + 1, doc_title))
    print('______________________\n')


if __name__ == "__main__":
    # 2.2.2 Modèle de recherche vectoriel

    # ---------- Choose here the model you want to apply for vectorial search ----------
    # - possible weightings for TF : w.tf, w.tf_binary, w.tf_id, w.tf_sqrt, w.tf_log, w.tf_log1p, w.tf_norm
    # - possible weightings for IDF : w.idf, w.idf_unary, w.idf_log, w.idf_smooth, w.idf_proba
    # - possible similarity measures for RSV : w.rsv_cos, w.rsv_dice, w.rsv_jaccard, w.rsv_overlap
    # Some weightings can take customised params like this :
    # - w.custom(w.tf_log, const=c, base=b), w.custom(w.tf_norm, k=k),  w.custom(w.idf_log, const=c, base=b)
    tf_ = w.tf
    idf_ = w.idf
    rsv_ = w.rsv_cos

    # ---------- Simulations ----------
    print("2.2.1 Vectorial Search Model\n")
    print("--- Collection CACM : Requêtes prédéfinies dans query.text ---")
    queries = read_queries('query.text')
    for query in queries:
        query_tokens = Collection(None).process(query)
        display_query(query_tokens)
        result = search_for_query(query_tokens, FreqIndex('CACM'), tf=tf_, idf=idf_, rsv=rsv_)
        display_result(result)
    print("")

    print("--- Collection CACM : Saisie en ligne de commande ---")
    start_search_engine_cli('CACM', tf=tf_, idf=idf_, rsv=rsv_)

    print("--- Collection CS276 : Saisie en ligne de commande ---")
    start_search_engine_cli('CS276', tf=tf_, idf=idf_, rsv=rsv_)


    print("\n\n2.3 Vectorial Search in Compressed Index\n")
    from src.compression.index_readers import FreqIndexVBE

    print("--- Collection CS276 avec VBE Index : Saisie en ligne de commande ---")
    specific_index = FreqIndexVBE('CS276')
    start_search_engine_cli(None, specific_index)
