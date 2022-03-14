from openie import StanfordOpenIE
import spacy
import json
from stanfordcorenlp import StanfordCoreNLP
import re

def open_corpus ():
    with open('corpus/example-chat.txt', encoding='utf8') as r:
        corpus = r.read().replace('\n', ' ').replace('\r', '')
    return corpus

def list_to_string(s):
    str1 = "" 
    for ele in s: 
        str1 += ele  
    return str1 

# Extracts chains of coreferences.
def coreference_resolution (text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001)
    coref_chains = nlp.coref(text)
    nlp.close()
    return coref_chains

# Annotates text to have same reference for objects in same coreference chain.
def coreference_integration (text):
    text_without_symbols = text.replace("!", " ").replace("?", " ").replace(",", " ").replace(".", " ")
    coref_chains = coreference_resolution(text)
    for chain in coref_chains:
        for ref in chain:
            text_without_symbols = text_without_symbols.replace(ref[3], chain[0][3])
    return text_without_symbols

# Entity linking for the spn4re example triples.
def entity_linking (text, triples):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities
    seen_entities = list()

    for entity in all_linked_entities:
        if str(entity.get_id()) not in seen_entities:
            new_name = entity.get_label() + "_Q" + str(entity.get_id())
            triples = triples.replace(str(entity.get_span()), new_name)
            seen_entities.append(str(entity.get_id()))
    return triples

# Quick fix to remove duplicate relations.
def remove_duplicates (text):
    lst = text.split(" . ")
    lst = list(dict.fromkeys(lst))
    temp = ""
    for element in lst:
        temp += element + " . "
    return temp

# Overall architecture, specialised for the example conversation with SPN4RE
def combined_joint (text, triples, optional_coreference):
    annotated_text = text
    if(optional_coreference == True):
        annotated_text = coreference_integration(text)

    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        triples_linked = entity_linking(text, triples)
        triples_linked = remove_duplicates(triples_linked)
        print(triples_linked)

        graph_image = 'graph.png'
        # Find a better tool for generating graphs
        client.generate_graphviz_graph(triples_linked, graph_image)
        print('Graph generated: %s.' % graph_image)

if __name__ == '__main__':
    data = open_corpus()
    optional_coreference = True
    
    # Example conversation with SPN4RE applied.
    with open('spn4re-coref.txt') as f:
            temp = f.readlines()
            temp = list_to_string(temp)

    combined_joint(data, temp, optional_coreference)