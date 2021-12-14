from openie import StanfordOpenIE
import spacy
import json
from stanfordcorenlp import StanfordCoreNLP
import re

def list_to_string(s):
    str1 = "" 
    for ele in s: 
        str1 += ele  
    return str1 

# Converts list of dictionaries to a string.
def triple_list_to_string(s): 
    str1 = "" 
    for ele in s:
        test = list(ele.values())
        for result in test:
            str1 += (result + " ")
            # entity_linking(str1)
    return str1 

def open_corpus ():
    with open('corpus/example-chat.txt', encoding='utf8') as r:
        corpus = r.read().replace('\n', ' ').replace('\r', '')
    return corpus

# Prints corpus of triples.
def print_triple_corpus (triple_corpus):
    print('Found %s triples in the corpus.' % len(triples_corpus))
    for triple in triples_corpus[:3]:
        print('|-', triple)
    print('[...]')

# Generates a graph using Graphviz.
def openie_graph (triple_corpus):
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        graph_image = 'graph.png'
        temp = triple_list_to_string(triple_corpus)
        client.generate_graphviz_graph(temp, graph_image)
        print('Graph generated: %s.' % graph_image)

# Extracts triples using the Stanford OpenIE library.
def openie_extract_triples (text):
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        corpus = text
        triple_corpus = client.annotate(corpus)
    return triple_corpus

# Extracts chains of coreferences.
def coreference_resolution (text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001) # Should update code to only use this not openie lib.
    # sentence = 'Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008.'
    coref_chains = nlp.coref(text)
    nlp.close()
    return coref_chains

# Extracts chains of coreferences.
def triple_integration (text):
    li = text.replace("!", " ").replace("?", " ").replace(",", " ").replace(".", " ")
    coref_chains = coreference_resolution(text)
    for chain in coref_chains:
        for ref in chain:
            li = li.replace(ref[3], chain[0][3])
    
    # Extracts triples on the text annotated with coreference chains.
    triples = openie_extract_triples(li)

    return triples

# Performs entity linking using the Spacy library.
def entity_linking (text):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities
    for entity in all_linked_entities:
        entity.pretty_print()

# Performs entity linking using the Spacy library on one entity.
def entity_linking_single (text):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities
    for entity in all_linked_entities:
        if entity != None:
            strins = entity.get_label() + " Q" + str(entity.get_id())
            print(strins)
            return strins
    return None

# Iterates over entities in triple corpus replacing them with wikidata entities.
def entity_linking_iterative (triple_corpus):
    for triple in triple_corpus:
        sbj = entity_linking_single(triple['subject'])
        if sbj != None:
            triple['subject'] = sbj
        obj = entity_linking_single(triple['object'])
        if obj != None:
            triple['object'] = obj
    return triple_corpus

# Quick fix to remove duplicate entities before entity linking.
def el_no_duplicates (triple_corpus):
    text = triple_list_to_string(triple_corpus)
    li = list(text.split(" "))
    li = list(dict.fromkeys(li))
    for l in li:
        entity_linking(l)

# Combined method containing coreference resolution, triple extraction and entity linking with graph generation.
def combined (text):
    li = text.replace("!", " ").replace("?", " ").replace(",", " ").replace(".", " ")
    coref_chains = coreference_resolution(text)
    for chain in coref_chains:
        for ref in chain:
            li = li.replace(ref[3], chain[0][3])

    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        triple_corpus = client.annotate(li)
        triple_corpus = entity_linking_iterative(triple_corpus)       

        graph_image = 'graph.png'
        temp = triple_list_to_string(triple_corpus)
        client.generate_graphviz_graph(temp, graph_image)
        print('Graph generated: %s.' % graph_image)



if __name__ == '__main__':
    #triple_corpus = extract_triples()
    # print_triple_corpus(triple_corpus)
    data = open_corpus()
    #triple_integration(data)
    #text = triple_list_to_string(corp)
    #print (corp)
    #corp = open_corpus()
    combined(data)
    #coreference_resolution(corp)
    # print(text)
    #triples = triple_integration(data)
    #for triple in triples:
        #sbj = entity_linking_single(triple['subject'])
        #if sbj != None:
            #triple['subject'] = sbj
        #obj = entity_linking_single(triple['object'])
        #if obj != None:
            #triple['subject'] = obj

    #print(triples)
