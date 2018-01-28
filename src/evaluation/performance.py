import time
import os
import statistics
import matplotlib.pyplot as plt
from config import RES_DIR

from src.language_processing.processing import Collection, CACM
from src.indexing.doc_index import DocBSBI
from src.indexing.freq_index import FreqBSBI
from src.searching import bool_search as bool, vect_search as vect
from src.searching.index_reader import DocIDIndex, FreqIndex


def timeit(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


def display_execution_times(labels, times):
    for i, label in enumerate(labels):
        print("- %s: %.3f (sec)" % (label.replace('\n', ' '), times[i]))
    print("")


def display_memory_sizes(labels, sizes):
    for i, label in enumerate(labels):
        print("- %s: %.0f (bytes)" % (label, sizes[i]))


def perf_processing():
    """ Temps traitement linguistique des collections """
    print("----- Test language processing -----")
    perf_half_cacm = []
    perf_cacm = []

    processes = [{'tokn': False, 'filt': False, 'norm': False},
                 {'tokn': True, 'filt': False, 'norm': False},
                 {'tokn': False, 'filt': True, 'norm': False},
                 {'tokn': False, 'filt': False, 'norm': True},
                 {'tokn': True, 'filt': True, 'norm': False},
                 {'tokn': False, 'filt': True, 'norm': True},
                 {'tokn': True, 'filt': True, 'norm': True}]

    # Pour la moitié de CACM (1602 documents)
    print("For half of CACM...")
    for params in processes:
        cacm = Collection(CACM(), **params)
        perf_half_cacm.append(timeit(cacm.process_collection, 0.5))

    # Pour tout CACM (3204 documents)
    print("For CACM...")
    for params in processes:
        cacm = Collection(CACM(), **params)
        perf_cacm.append(timeit(cacm.process_collection))

    return perf_half_cacm, perf_cacm


def perf_indexing():
    """ Temps de calcul pour l'indexation """
    print("----- Test indexing -----")

    # Pour la création d'un index docID
    cacm_docid = DocBSBI(Collection(CACM()))
    perf_docid = timeit(cacm_docid.construct_index)

    # Pour la création d'un index de fréquences
    cacm_freq = FreqBSBI(Collection(CACM()))
    perf_freq = timeit(cacm_freq.construct_index)

    return perf_docid, perf_freq


def perf_searching():
    """ Temps de réponse à une requête """
    print("----- Test query searching -----")
    perf_bool = []
    perf_vect = []

    # Pour une recherche booléenne
    print("Boolean search...")
    queries = bool.read_queries('query_bool.text')
    index = DocIDIndex('CACM')
    for q in queries:
        time = timeit(bool.search_for_query, q, index)
        perf_bool.append((sum(map(len, q)), time))

    # Pour une recherche vectorielle
    print("Vectorial search...")
    def search_query(query, index):
        query_tokens = Collection(None).process(query)
        vect.search_for_query(query_tokens, index)

    queries = vect.read_queries('query.text')
    index = FreqIndex('CACM')
    for q in queries:
        time = timeit(search_query, q, index)
        perf_vect.append((len(q.split()), time))

    return perf_bool, perf_vect


def perf_storing():
    """ Occupation de l’espace disque par les différents index """
    space_docid = {}
    space_freq = {}

    # Pour l'index DocID
    folder_docid = os.path.join(RES_DIR, 'Index_DocID', 'CACM')
    for file in os.listdir(folder_docid):
        size = os.path.getsize(os.path.join(folder_docid, file))
        space_docid[file] = size

    # Pour l'index de fréquences
    folder_freq = os.path.join(RES_DIR, 'Index_Freq', 'CACM')
    for file in os.listdir(folder_freq):
        size = os.path.getsize(os.path.join(folder_freq, file))
        space_freq[file] = size

    return space_docid, space_freq


if __name__ == "__main__":
    # 2.3.1 Evaluation de la performance pour la collection CACM
    print("2.3.1 Evaluation of performance for CACM\n")

    # Loading collections and language processing
    half_cacm, cacm = perf_processing()

    plt.figure(1, figsize=(12,6))
    plt.title("Chargement et traitement linguistique des collections")
    plt.xlabel("(1) half CACM = 1602 docs, (2) CACM = 3204 docs")
    plt.ylabel("temps d'exécution (s)")

    x1 = range(0, 28, 4)
    x2 = [x + 1.6 for x in x1]
    labels = ["None", "Tokenize", "Filter", "Stemming", "Tokenize\n+ Filter", "Filter\n+ Stemming", "All"]
    plt.bar(x1, half_cacm, width=1.6, color='c', align='edge')
    plt.bar(x2, cacm, width=1.6, color='b', align='edge', tick_label=labels)

    print("\nExecution times for language processing with CACM:")
    display_execution_times(labels, cacm)

    # Indexing collections
    docid, freq = perf_indexing()
    print("\nCalculation time for indexing CACM:")
    display_execution_times(["DocID index", "Frequency index"], [docid, freq])

    # Responding to a query
    perf_bool, perf_vect = perf_searching()
    len_queries_b, times_b = zip(*perf_bool)
    len_queries_v, times_v = zip(*perf_vect)

    ### Boolean search
    plt.figure(2)
    plt.title("Temps de réponse à une requête booléenne")
    plt.xlabel("nombre d'opérandes dans la requête")
    plt.ylabel("temps d'exécution (s)")
    plt.plot(len_queries_b, times_b, 'g.')

    print("\nResponse time for boolean queries in query_bool.text:")
    print("- Average time:", statistics.mean(times_b), "(sec)")
    print("- Median time:", statistics.median(times_b), "(sec)")

    plt.figure(3)
    plt.title("Temps de réponse à une requête booléenne")
    plt.xlabel("temps d'exécution (s)")
    plt.ylabel("répartition")
    plt.hist(times_b, 30)

    ### Vectorial search
    plt.figure(4)
    plt.title("Temps de réponse à une requête vectorielle")
    plt.xlabel("nombre de mots dans la requête")
    plt.ylabel("temps d'exécution (s)")
    plt.plot(len_queries_v, times_v, 'g.')

    print("\nResponse time for vectorial queries in query.text:")
    print("- Average time:", statistics.mean(times_v), "(sec)")
    print("- Median time:", statistics.median(times_v), "(sec)")

    plt.figure(5)
    plt.title("Temps de réponse à une requête vectorielle")
    plt.xlabel("temps d'exécution (s)")
    plt.ylabel("répartition")
    plt.hist(times_v, 30)

    # Storing index in memory
    space_docid, space_freq = perf_storing()
    files_d, sizes_d = list(space_docid.keys()), list(space_docid.values())
    files_f, sizes_f = list(space_freq.keys()), list(space_freq.values())

    print("\nSpace occupied by CACM DocID indexes in memory:")
    display_memory_sizes(files_d, sizes_d)

    print("\nSpace occupied by CACM Freq indexes in memory:")
    display_memory_sizes(files_f, sizes_f)

    plt.figure(6, figsize=(9, 5))
    plt.title("Space occupied by CACM indexes in memory")
    plt.ylabel("space (kB)")

    all_files = list(set(files_d) | set(files_f))
    x3 = range(0, 3 * len(all_files), 3)
    x4 = range(1, 1 + 3 * len(all_files), 3)
    for f in all_files:
        if f not in space_docid:
            space_docid[f] = 0
        if f not in space_freq:
            space_freq[f] = 0

    sizes_d_ = [space_docid.get(f)/1024 for f in all_files]
    sizes_f_ = [space_freq.get(f)/1024 for f in all_files]

    plt.bar(x3, sizes_d_, width=0.9, color='y', align='edge', label='DocID index')
    plt.bar(x4, sizes_f_, width=0.9, color='g', align='edge', label='Frequency index', tick_label=all_files)
    plt.legend()

    # Show graphs
    plt.show(1)
    plt.show(2)
    plt.show(3)
    plt.show(4)
    plt.show(5)
    plt.show(6)

