from src.searching.index_reader import DocIDIndex
from config import QUERIES_DIR
import os


class WrongFormatError(Exception): pass


# Boolean queries will be represented in Normal Disjonctive Form
def read_queries(query_file='query_bool.text'):
    """ Read boolean queries that are written on query_file """
    queries = list()

    query_path = os.path.join(QUERIES_DIR, query_file)
    with open(query_path, 'r') as f:
        line = f.readline()
        query = []

        while line:
            if line.startswith('.I'):
                nb_literals = int(line[3:])
                for n in range(nb_literals):
                    line = f.readline()
                    literal = line.split()
                    if len(literal) == 0:
                        raise WrongFormatError
                    query.append(literal)
                queries.append(query)
                query = []
            else:
                line = f.readline()
    return queries


def start_search_engine_cli(collection_name, index=None):
    """ Input queries in Command Line Interface and search for results immediately after each request """
    print('Entrez en ligne de commande les requêtes sous forme normale disjonctive (FND)')
    print("Chaque clause conjonctive sera indiquée sur une ligne à l'aide d'une suite de mots espacés")
    print("Pour indiquer une négation, précedez le mot d'un signe - non espacé")
    print('______________________\n')

    running = True
    if not index:
        index = DocIDIndex(collection_name)
    while running:
        query = input_query()
        display_query(query)
        result = search_for_query(query, index)
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
    """ Write boolean query in command line """
    print("\n---------- Nouvelle requête ----------")

    query = []
    literals_count = 1
    while True:
        print("Clause conjonctive n°", literals_count, ':')
        line = input()
        if len(line) > 0:
            query.append(line.split())
        print("Voulez vous ajouter une nouvelle clause alternative ? (Y/N)")
        answer = input().lower()
        while not answer in ['y', 'n']:
            print("Entrée invalide. Voulez vous ajouter une nouvelle clause ? (Y/N)")
            answer = input().lower()
        if answer == 'n':
            break
        literals_count += 1
    return query


def display_query(query):
    """ Display in console the boolean query shaped in human readable style """

    def format_variable(word):
        return "NOT %s " % word[1:] if word.startswith('-') else word + " "

    def format_literal(literal, parenthesis):
        literal_str = format_variable(literal[0])
        for word in literal[1:]:
            literal_str += "AND " + format_variable(word)
        return "( " + literal_str + ") " if parenthesis else literal_str

    query_str = format_literal(query[0], len(query) > 1 and len(query[0]) > 1)
    for literal in query[1:]:
        query_str += "OR " + format_literal(literal, len(literal) > 1)

    print("\nRequête :", query_str)
    print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n')


def search_for_query(query, index):
    """ Run a boolean search in index for given query """
    relevant_docs = set()

    for clause in query:
        clause = sorted(clause, key=lambda x: 1 if x.startswith('-') else 0)

        first_term = clause[0]
        if not first_term.startswith('-'):
            docs = index.find_documents(first_term.lower())
        else:
            docs = index.get_all_documents() - index.find_documents(first_term[1:].lower())

        for term in clause[1:]:
            if len(docs) == 0:
                break
            if not term.startswith('-'):
                docs = docs & index.find_documents(term.lower())
            else:
                docs = docs - index.find_documents(term[1:].lower())

        relevant_docs = relevant_docs | docs

    return index.get_documents_from_ids(sorted(relevant_docs))


def display_result(list):
    """ Display list of results in console (they are ordered by their id) """
    print("Liste des résultats :")
    if len(list) == 0:
        print("Aucun résultat")
    for rank, doc_title in enumerate(list):
        print("%i) %s" % (rank + 1, doc_title))
    print('______________________\n')


if __name__ == "__main__":
    # 2.2.1 Modèle de recherche booléen

    print("2.2.1 Boolean Search Model\n")
    print("--- Collection CACM : Requêtes prédéfinies dans query_bool.text ---")
    queries = read_queries('query_bool.text')
    for query in queries:
        display_query(query)
        result = search_for_query(query, DocIDIndex('CACM'))
        display_result(result)
    print("")

    print("--- Collection CACM : Saisie en ligne de commande ---")
    start_search_engine_cli('CACM')

    print("--- Collection CS276 : Saisie en ligne de commande ---")
    start_search_engine_cli('CS276')


    print("\n\n2.3 Boolean Search in Compressed Index\n")
    from src.compression.index_readers import DocIDIndexVBE

    print("--- Collection CS276 avec VBE Index : Saisie en ligne de commande ---")
    specific_index = DocIDIndexVBE('CS276')
    start_search_engine_cli(None, specific_index)