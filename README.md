# Fondements de la Recherche d’Information-WEB
*Cours de 3A : Projet*

## Résumé des instructions pour l'exécution

Pour exécuter le projet il est nécessaire d'inclure les corpus de données étudiés (qui ne sont pas fournis à cause de leur taille).
Pour cela, deux solutions sont possibles :
* soit vous téléchargez et copiez les deux dossiers CACM et CS276 tels qu'ils se présentent sur Claroline dans le répertoire res/Data
* soit vous modifiez le chemin DATA_DIR dans le fichier .env qui se trouve à la racine pour indiquer l'emplacement du dossier qui contient CACM et CS276 sur votre machine

Avant toute chose, ouvrez un terminal à la racine du projet et exécutez les deux commandes suivantes:
* `pip install -r requirements.txt`
* ``export PYTHONPATH=$PYTHONPATH:`pwd` `` (si vous souhaitez lancer le projet depuis une console)

Les fichiers exécutables sont :
* _src/language_processing/processing.py_
* _src/indexing/index_builder.py_ (étape nécessaire pour certaines autres exécutions)
* _src/searching/bool_search.py_ (*)
* _src/searching/vect_search.py_ (*)
* _src/evaluation/performance.py_
* _src/evaluation/pertinence.py_ (*)
* _src/compression/vb_encoding.py_

(*) L'exécution de ces fichiers nécessite au préalable l'exécution de _src/indexing/index_builder.py_ qui va créer les indexs pour chaque collection.

## 1. Architecture du projet, corpus de données
Le dossier présent s'articule autour de 2 répertoires _src_ et _res_, le premier contenant le code source du projet, et le deuxième les données (collections, indexes, requêtes, jugements de pertinence...) sur lesquelles j'ai travaillé.

Autant le dossier _src_ est personnel et nécessaire pour faire tourner le projet (ainsi que le fichier _config_ à la racine), autant le dossier des ressources _res_ peut être reconstruit en y plaçant correctement les données des collections étudiées (la collection CACM et le corpus issu du cours CS 276 de l’Université de Stanford), ce qui est sans doute préférable à un téléchargement complet vu la taille du dossier.

### Les ressources _res_

**Le répertoire des ressources doit contenir obligatoirement les sous-dossier suivants:**
* Un dossier __Data__ correspondant aux données brutes disponibles et téléchargeables sur Claroline. Il comprend lui même un sous-dossier pour chaque corpus utilisé.
  * CACM : contient les fichiers _cacm.all_, _common-words_, _query.text_ et _qrels.text_
  * CS276 : contient 98998 documents répartis dans 10 sous répertoires numérotés de 0 à 9
* Un dossier __Queries__ qui rassemble les fichiers des requêtes prédéfinies. En l'occurence j'en ai placé deux, mais il est possible d'en créer d'autres sur le même schéma.
  * _query.text_ : C'est une copie du fichier _query.text_ de CACM que j'ai simplement déplacé ici
  * _query_bool.text_ : Ce fichier est une traduction des 64 requêtes de _query.text_ sous forme booléenne (lisible par le moteur de recherche booléen)

**Le répertoire des ressources peut également contenir les sous-dossiers suivants:**
* Index_DocID : Fichiers d'index de type DocID pour les 2 collections
* Index_Freq : Fichiers d'index de type Frequency pour les 2 collections
* IndexVBE_DocID : Fichiers d'index compressés de type DocID pour la collection CS276
* IndexVBE_Freq : Fichiers d'index compressés de type Frequency pour la collection CS276

Ceux-ci sont nécessaire pour exécuter les recherches (_searching.bool_search_, _searching.vect_search_) et pour l'évaluation du systeme (_evaluation.performance_, _evaluation.pertinence_) mais ils sont générés automatiquement lorsque l'on exécute le fichier _indexing.index_builder_)

**Les emplacements respectifs du dossier des données, du dossier des ressources (où seront générés les indexes) et du dossier des requêtes peuvent être redéfinis dans le fichier .env à la racine.** Il est donc tout à fait possible de récupérer ou de génerer ces données à d'autres endroits à condition d'en modifier le chemin dans _.env_

### Le code source _src_

Le code source s'articule autour de 6 parties :
* Un fichier __interface.py__ qui permet de charger en mémoire les données initiales du dossier __Data__
* Un dossier __language_processing__ qui effectue les traitements linguistiques sur les documents bruts (2.1)
* Un dossier __indexing__ qui construit et stocke pour chaque collection les structures de données utiles à la recherche, comme l'index inversé, les dictionnaires de termes et de documents... (2.2.0)
* Un dossier __searching__ qui met en place les modèles de recherche booléens et vectoriels (2.2.1 et 2.2.2)
* Un dossier __evaluation__ qui établit des mesures de performance et de pertinence pour les deux systèmes de recherche (booléen et vectoriel) appliqués à la collection CACM (2.3)
* Un dossier __compression__ qui met en œuvre la méthode de compression Variable Byte Encoding pour générer l'index inversé de la collection CS276 (3.0)

## 2. Mode d'emploi détaillé et explications
Les démarches à effectuer pour exécuter les différentes parties du projet et les explications sur les choix d'implémentation seront données dans l'ordre de l'énoncé

### Tâche 1 : Création d’un index inversé et moteur de recherche booléen et vectoriel
#### 2.1 Traitements linguistiques
Les traitements linguistiques (Tokenization, Filtrage, Normalisation par troncature) sont réalisés dans le dossier _language_processing_.

Il est possible d'ajuster les traitements qui seront effectués sur les documents en jouant sur les paramêtres `tokn`, `filt` et `norm` lors de l'instanciation d'un objet Collection.

**Tokenization** : Il s'agit de séparer le texte en une séquence de mots. Comme préconisé dans l'énoncé, j'ai procédé de façon simplifiée étant donné la nature des collections, c'est-à-dire en considérant tous les espaces et les caractères non alpha-numériques comme des séparateurs. Cette méthode est beaucoup plus rapide que _nltk.word_tokenize_, que je n'ai donc pas utilisé pour cette raison, bien que ce dernier soit un peu plus précis. La tokenisation décrite ci-dessus n'est effectuée que si `tokn = True` (choix par défaut), sinon on se contente de séparer les mots déjà pré-espacés en enlevant juste les 's (c'est le cas de CS276)

**Filtrage** : Je vérifie que les tokens trouvés sont bien alphanumériques et je les compare avec une stop-liste pour ne garder que les plus significatifs. Pour ce faire, j'ai utilisé la même stop-liste pour les deux collection (celle de _common-words_ dans CACM). Cette étape n'est effectuée que si `filt=True` (par défaut True)

**Normalisation** : Une troncature (ou Stemming) est effectuée grâce à l'algorithme _PorterStemmer_ implémenté dans gensim (_gensim.parsing.porter.PorterStemmer_). J'ai utilisé cet algorithme car il est plus rapide que celui de NLTK (_nltk.stem.PorterStemmer_). Néanmoins, la normalization prend un peu de temps et pourra être négligée pour la suite. C'est pourquoi j'ai mis par défaut `norm=False`. Si malgé tout vous souhaitez tester tout le code avec Stemming (indexation, recherche, évaluation...) il vous faudra effectuer les deux étapes suivantes :
* Remplacer `Norm=False` par `Norm=True` dans le constructeur de Collection (processing l.15)
* Remplacer `stemming=False` par `stemming=True` dans la signature de _find_documents_ de la classe DocIDIndex dans le fichier searching.index_reader (l.56) : Cela permet de faire aussi le traitement de normalisation sur les opérandes des requêtes booléennes

**La réponse aux questions 1 à 5 sont générées lors de l'exécution du fichier _processing.py_**. En cas de problème, on pourra aussi voir les réponses et les graphes dans le fichier _results.pdf_ à la racine.

#### 2.2 Indexation
La construction et la sauvegarde des structures de données nécessaires à la recherche (index inversé, dictionnaire de termes, dictionnaire de documents et parfois index non inversé) sont effectuées par le code du dossier _indexing_

Deux types d'index inversés ont été construits, l'un étant plus adapté aux requêtes booléennes (DocID index), l'autre plus adapté aux requêtes vectorielles (Frequency index)
* DocID index : line = "term_id doc_id1 doc_id2 doc_id3..."
* Frequency index : line = "term_id nb_docs doc_id1:count1 doc_id2:count2 doc_id3:count3..."

Pour chaque collection (CACM et CS276), ces deux types d'index vont donc être construits dans des dossiers séparés _Index_DocID_ et _Index_Freq_. Dans _Index_DocID_, les dossiers d'index des collections comprendront un index inversé _index.txt_ et deux dictionnaires _documents.txt_ et _terms.txt_ qui font la correspondance (nom,id) des documents et des terms. Dans _Index_Freq_, on aura en plus un index non inversé _doc_index.txt_ qui facilitera la recherche vectorielle.

**Pour lancer la construction des index, il suffit d'exécuter le fichier _index_builder.py_**. Attention, l'exécution est longue (plusieurs minutes) et détruira les fichiers qui préexistaient dans les dossiers _Index_DocID_ et _Index_Freq_. En inspectant le _main_, vous pourrez voir qu'il y a en fait 6 constructions lancées successivement (pour chaque collection, pour chaque type d'index + 2 index compressés pour CS276 comme demandé en 3.0). Vous pouvez restreindre les constructions en commentant les autres.

#### 2.2.1 Modèle de recherche booléen
Le modèle de recherche booléen est mis en place **dans le fichier _bool_search.py_ du dossier _searching_**. Pour lancer la recherche, il faut donc exécuter ce fichier.

Les requêtes booléennes devront être formulées sous forme normale disjonctive (FND). Plus concrêtement, il y a deux manières de lire les requêtes booléennes qui sont implémentées. La première consiste à décrire les requêtes dans un fichier placé dans le dossier _Queries_ des ressources. La deuxième consiste à entrer les requêtes une à une en ligne de commande grâce à un parcours guidé.

Pour les requêtes prédéfinies dans un fichier, comme par exemple dans _query_bool.text_:
* Chaque nouvelle requette est introduite par `.I` suivi du nombre de clauses conjonctives
* Chaque clause conjonctive est décrite comme une suite de mots espacés
* Pour spécifier une négation, faire précéder le mot du signe `-` non espacé

#### 2.2.2 Modèle de recherche vectoriel
Le modèle de recherche vectoriel est mis en place **dans le fichier _vect_search.py_ du dossier _searching_**. Pour lancer la recherche, il faut donc exécuter ce fichier.

Le requêtes vectorielles se présentent comme de petits textes formulés de manière libre. La requête sera indexée et traitée comme le sont les documents. Là encore, il y a deux façons de lire une requête vectorielle, soit à partir d'un fichier comme query.text (seuls les contenus des champs `.W` seront lus), soit directement en ligne de commande.

Enfin, plusieurs méthodes de pondération semblables à tf-idf ont été implémentées, ainsi que plusieurs mesures de similarité pour la comparaison. Tous ces paramêtres sont définis sous forme de fonctions de même signature dans le fichier _weightings.py_. Les ajustements du modèle peuvent se faire en modifiant les paramêtres `tf_`, `idf_` et `rsv_` au début du _main_ de _vect_search.py_ avant de lancer les recherches.

#### 2.3 Evaluation pour la collection CACM
Les outils d'évaluation du système sont définis dans le dossier _evaluation_, qui comprend des mesures de performance et des mesures de pertinence.

**Exécutez les fichiers _performance.py_ et _pertinence.py_ pour lancer les évaluations sur la collection CACM**
En cas de problème, les résultats et les graphs pourront être visualisés dans le fichier _results.pdf_ à la racine.

L'évaluation a été effectuée sans normalisation des documents et des requêtes puis avec normalisation pour comparer. Malgré un temps de construction des index significativement plus longs, on observe un espace mémoire occupé moins important, des requêtes légèrement plus rapides et surtout une pertinence accrue.

### Tâche 2 : Création d’un index inversé compressé et moteur de recherche booléen et vectoriel
La méthode de compression Variable Byte Coding est implémentée dans le dossier _compression_ et mise en oeuvre au moment de la création et de la lecture des deux types d'index inversé _index.txt_ et de l'index normal _doc_index.txt_

**Un exemple d'encodage et de décodage est donné par l'exécution du fichier _vb_encoding.py_** Cette nouvelle méthode d'écriture et de lecture des index est testée sur la collection CS276 et peut être observée lors de l'exécution des fichiers _indexing.index_builder.py_, _searching.bool_search.py_ et _searching.vect_search.py_ précédemment cités.
