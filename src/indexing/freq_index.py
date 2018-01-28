from multiprocessing import Pool
from collections import defaultdict
from threading import Thread
import itertools
import operator
import os
from config import RES_DIR

from src.indexing.index_builder import BSBI, MapReduce


class FreqBSBI(BSBI, MapReduce):
    """ BSBI algorithm for constructing Frequency Indexes with Map Reduce approach, useful for vectorial requests """

    def __init__(self, collection):
        BSBI.__init__(self, collection, 'Freq')
        MapReduce.__init__(self)

    # Map Reduce methods
    def map(self, doc_name, tokens):
        # key = doc_name, value = tokens
        pairs = list()
        doc_id = self.look_for_document(doc_name)
        for term in tokens:
            term_id = self.look_for_term(term)
            pairs.append((term_id, doc_id))
        return self.combine(pairs)

    def combine(self, pairs):
        sorted_pairs = sorted(pairs)
        grouped_pairs = list()

        iter = itertools.groupby(sorted_pairs, operator.itemgetter(0))
        for key, group in iter:
            values = [item[1] for item in group]
            grouped_pairs.append((key, values))

        return grouped_pairs

    def shuffle_sort(self, all_pairs):
        sorted_pairs = sorted(all_pairs)

        # Write current scanned documents index to disk
        docs_dict = defaultdict(dict)
        for term, docs in list(all_pairs):
            docs_dict[docs[0]][term] = len(docs)
        write_docs = Thread(target=self.add_block_to_disk, args=(docs_dict, 'doc_index'))
        write_docs.start()

        all_values = list()
        iter = itertools.groupby(sorted_pairs, operator.itemgetter(0))
        for key, group in iter:
            values = itertools.chain(*[item[1] for item in group])
            all_values.append((key, values))

        write_docs.join()
        pool = Pool()
        return list(pool.starmap(self.__class__.reduce, all_values))

    @staticmethod
    def reduce(term_id, documents):
        doc_dict = dict()
        for doc_id, doc_group in itertools.groupby(documents):
            doc_dict[doc_id] = len(list(doc_group))
        return term_id, doc_dict

    # BSBI methods
    def parse_block(self, block_name):
        all_lists_pairs = self.collection.process_block(block_name, self.map)
        return list(itertools.chain(*all_lists_pairs))

    def invert_block(self, pairs):
        postings = dict(self.shuffle_sort(pairs))
        return postings

    def write_block_to_disk(self, postings, block_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '.txt')
        with open(path, "w") as file:
            for term_id, documents in sorted(postings.items()):
                doc_list = ['%i:%i' % (doc_id, freq) for doc_id, freq in sorted(documents.items())]
                file.write(' '.join([str(term_id)] + [str(len(doc_list))] + doc_list) + '\n')

    def add_block_to_disk(self, postings, file_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, file_name + '.txt')
        with open(path, "a") as file:
            for doc_id, terms in sorted(postings.items()):
                term_list = ['%i:%i' % (term_id, freq) for term_id, freq in sorted(terms.items())]
                file.write(' '.join([str(doc_id)] + [str(len(term_list))] + term_list) + '\n')

    def merge_blocks(self, blocks, final_file):
        indexes = list()

        # Read all blocks
        for block_name in blocks:
            path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '.txt')
            index = dict()
            with open(path, "r") as file:
                for line in file:
                    ids = line.split()
                    index[int(ids[0])] = {int(doc.split(':')[0]): int(doc.split(':')[1]) for doc in ids[2:]}
            indexes.append(index)
            os.remove(path)

        # Merge postings list in memory
        term_ids = set().union(*indexes)
        global_index = dict()
        for term_id in term_ids:
            global_docs = dict()
            list_docs = [index.get(term_id, {}) for index in indexes]
            for doc_id in set().union(*list_docs):
                global_docs[doc_id] = sum(docs.get(doc_id, 0) for docs in list_docs)
            global_index[term_id] = global_docs

        # Write result to disk
        self.write_block_to_disk(global_index, final_file)
