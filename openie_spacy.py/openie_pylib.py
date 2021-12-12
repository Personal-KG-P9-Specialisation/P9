from openie import StanfordOpenIE
import spacy
import json
from stanfordcorenlp import StanfordCoreNLP

# Converts list of dictionaries to a string.
def listToString(s): 
    str1 = "" 
    for ele in s:
        test = list(ele.values())
        for result in test:
            str1 += (result + " ")
            # entity_linking(str1)
    return str1 

def openCorpus ():
    with open('corpus/example-chat.txt', encoding='utf8') as r:
        corpus = r.read().replace('\n', ' ').replace('\r', '')
    return corpus

# Prints corpus of triples.
def printCorpus (triple_corpus):
    print('Found %s triples in the corpus.' % len(triples_corpus))
    for triple in triples_corpus[:3]:
        print('|-', triple)
    print('[...]')

# Generates a graph using Graphviz.
def generate_graph (client, corpus):
    graph_image = 'graph.png'
    client.generate_graphviz_graph(corpus, graph_image)
    print('Graph generated: %s.' % graph_image)

# Extracts triples using the Stanford OpenIE library.
def extract_triples ():
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        corpus = openCorpus()
        triples_corpus = client.annotate(corpus)
        # generate_graph(client, corpus)
    return triple_corpus

# Extracts chains of coreferences.
def coreference_resolution (text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001) # Should update code to only use this not openie lib.
    # sentence = 'Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008.'
    print('Tokenize:', nlp.coref(text))
    nlp.close()

# Performs entity linking using the Spacy library.
def entity_linking (text):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities
    for entity in all_linked_entities:
        entity.pretty_print()

# Quick fix to remove duplicate entities before entity linking.
def el_no_duplicates (triple_corpus):
    text = listToString(triple_corpus)
    li = list(text.split(" "))
    li = list(dict.fromkeys(li))
    for l in li:
        entity_linking(l)

if __name__ == '__main__':
    # triple_corpus = extract_triples()
    # printCorpus(triple_corpus)
    # text = listToString(triple_corpus)
    corp = openCorpus()
    
    coreference_resolution(corp)
    # print(text)
    # entity_linking(text)
