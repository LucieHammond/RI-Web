from multiprocessing import Pool
import itertools
import operator
import os
from config import RES_DIR

from src.indexing.index_builder import BSBI, MapReduce


class DocBSBI(BSBI, MapReduce):
    """ BSBI algorithm for constructing DocID Indexes with Map Reduce approach, useful for boolean requests """

    def __init__(self, collection):
        BSBI.__init__(self, collection, 'DocID')
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
        return sorted(set(pairs))

    def shuffle_sort(self, all_pairs):
        sorted_pairs = sorted(all_pairs)
        all_values = list()

        iter = itertools.groupby(sorted_pairs, operator.itemgetter(0))
        for key, group in iter:
            values = [item[1] for item in group]
            all_values.append((key, values))

        pool = Pool()
        return list(pool.starmap(self.__class__.reduce, all_values))

    @staticmethod
    def reduce(term_id, documents):
        return term_id, sorted(documents)

    # BSBI methods
    def parse_block(self, block_name):
        list_pairs = self.collection.process_block(block_name, self.map)
        return set().union(*list_pairs)

    def invert_block(self, pairs):
        postings = dict(self.shuffle_sort(pairs))
        return postings

    def write_block_to_disk(self, postings, block_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '.txt')
        with open(path, "w") as file:
            for term_id, documents in sorted(postings.items()):
                file.write(' '.join(map(str,[term_id] + documents)) + '\n')

    def merge_blocks(self, blocks, final_file):
        indexes = list()

        # Read all blocks
        for block_name in blocks:
            path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '.txt')
            index = dict()
            with open(path, "r") as file:
                for line in file:
                    ids = list(map(int, line.split()))
                    index[ids[0]] = ids[1:]
            indexes.append(index)
            os.remove(path)

        # Merge postings list in memory
        term_ids = set().union(*indexes)
        global_index = dict()
        for term_id in term_ids:
            postings = set().union(*[set(index.get(term_id, [])) for index in indexes])
            global_index[term_id] = sorted(list(postings))

        # Write result to disk
        self.write_block_to_disk(global_index, final_file)
