from openie import StanfordOpenIE
import spacy
import json

# Converts list of dictionaries to a string.
def listToString(s): 
    str1 = "" 
    for ele in s:
        test = list(ele.values())
        for result in test:
            str1 += (result + " ")
    return str1 

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
        with open('corpus/example-chat.txt', encoding='utf8') as r:
            corpus = r.read().replace('\n', ' ').replace('\r', '')

        triples_corpus = client.annotate(corpus)

        # printCorpus(triple_corpus)
        # generate_graph(client, corpus)
    return triples_corpus

# Performs entity linking using the Spacy library.
def entity_linking (text):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities
    for entity in all_linked_entities:
        entity.pretty_print()


if __name__ == '__main__':
    triple_corpus = extract_triples()
    text = listToString(triple_corpus)
    print(text)
    entity_linking(text)
