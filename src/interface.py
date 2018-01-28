import os
import math
from threading import Thread
from collections import defaultdict
from config import DATA_DIR

def load_stop_words():
	""" We will take the same stop words for the 2 collections """
	path = os.path.join(DATA_DIR, 'CACM', 'common_words')
	stop_words = []

	with open(path, 'r') as f:
		for word in f:
			stop_words.append(word.replace('\n' ,''))

	return stop_words


def read_relevance_judgments():
	path = os.path.join(DATA_DIR, 'CACM', 'qrels.text')
	judgments = defaultdict(list)

	with open(path, 'r') as f:
		for line in f:
			nb_query, doc = map(int, line.split()[:2])
			judgments[nb_query].append(str(doc))
	return judgments


class CollectionLoader:

	def __init__(self, name):
		self._name = name
		self._blocks = []

	@property
	def name(self):
		return self._name

	@property
	def blocks(self):
		return self._blocks

	def load_all_documents(self, callback, percentage=1.0):
		"""
			:param queue : pipeline queue where we put new documents contents as soon as they are loaded
			:param percentage : The percentage of the collection we should read and return

			:return if no queue is given, return dict {doc_id: doc_text, ...}
		"""
		raise NotImplementedError

	def load_block(self, name, callback):
		raise NotImplementedError


class CACM(CollectionLoader):

	def __init__(self):
		CollectionLoader.__init__(self, 'CACM')
		self.markers = ['.I', '.T', '.W', '.B', '.A', '.N', '.X', '.K', '.C']
		self._blocks = ['all']

	def load_all_documents(self, callback, percentage=1.0, grouped=True):
		"""
			The documents will be picked regularly in the all collection for a better representativeness.
			For example, when we ask for half of the collection, we will pick one document out of 2 (the even ones)
		"""
		path = os.path.join(DATA_DIR, self.name, 'cacm.all')
		mod = 1/percentage  # We take one document out of mod (out of 2 if percentage is 50%)
		content = ""
		threads = list()

		def sender(doc_id, content):
			thread = Thread(target=callback, args=(doc_id, content))
			thread.start()
			threads.append(thread)

		with open(path, 'r') as f:
			doc_id = 0
			take_line = False

			for line in f:
				if take_line:
					if line[:2] not in self.markers:
						content += line
					else:
						take_line = False

				if line.startswith('.I'):
					if not grouped and content:
						sender(doc_id, content)
						content = ""
					doc_id = int(line[3:])
				elif line[:2] in ['.T', '.W', '.K'] and math.floor(doc_id % mod) == 0:
					take_line = True

			if not grouped and content:
				sender(doc_id, content)

		if not grouped:
			return threads
		thread = Thread(target=callback, args=(content, ))
		thread.start()
		return [thread]

	def load_block(self, name, callback):
		if name == 'all':
			return self.load_all_documents(callback, grouped=False)


class CS276(CollectionLoader):

	def __init__(self):
		CollectionLoader.__init__(self, 'CS276')
		self._blocks = [str(nb) for nb in range(10)]

	def load_all_documents(self, callback, percentage=1.0):
		"""
			Directories will not be separated when considering only a portion of the collection.
			The percentage will be used to determine how many directories of the collection we should read
		"""
		nb_dir = round(percentage * 10)
		threads = list()

		for nb in range(nb_dir):
			thread = self.load_block(str(nb), callback, grouped=True)
			threads.append(thread)

		return threads

	def load_block(self, name, callback, grouped=False):
		content = ""
		threads = list()

		path_dir = os.path.join(DATA_DIR, self.name, name)
		for file in os.listdir(path_dir):
			if not file.startswith('.'):
				with open(os.path.join(path_dir, file), 'r') as f:
					if grouped:
						content += f.read()
					else:
						doc_name = '%s_%s' % (name, file)
						thread = Thread(target=callback, args=(doc_name, f.read()))
						thread.start()
						threads.append(thread)

		if not grouped:
			return threads
		thread = Thread(target=callback, args=(content,))
		thread.start()
		return thread

	def load_document(self, nb_dir, name):
		path = os.path.join(DATA_DIR, self.name, str(nb_dir), name)
		with open(path, 'r') as f:
			return f.read()


if __name__ == "__main__":
	pass