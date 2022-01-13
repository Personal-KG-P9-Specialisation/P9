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

# Converts list of dictionaries to a string.
def triple_list_to_string(s): 
    str1 = "" 
    for ele in s:
        test = list(ele.values())
        for result in test:
            str1 += (result + " ")
            # entity_linking(str1)
        str1 += ". "
    return str1 

# Prints corpus of triples.
def print_triple_corpus (triple_corpus):
    print('Found %s triples in the corpus.' % len(triple_corpus))
    for triple in triple_corpus:
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

# Combines coreference resolution with triple extraction.
def triple_integration (text):
    # Annotates text with coreferences.
    annotated_text = coreference_integration(text)
    
    # Extracts triples on the text annotated with coreference chains.
    triples = openie_extract_triples(annotated_text)

    return triples

# Performs entity linking using the Spacy library on regular text.
def entity_linking (text, triple_corpus):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities

    # Replace entities in the extracted openIE triples with linked entities.
    for triple in triple_corpus:
        for entity in all_linked_entities:
            temp = str(entity.get_span())
            if triple['subject'] == temp:
                new_name = entity.get_label() + " Q" + str(entity.get_id())
                triple['subject'] = new_name
            if triple['object'] == temp:
                new_name = entity.get_label() + " Q" + str(entity.get_id())
                triple['object'] = new_name

    return triple_corpus

# Special entity linking for the spn4re example triples.
def entity_linking_spn4re (text, triples):
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

# Performs entity linking using the Spacy library on one entity.
def entity_linking_single (text):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(text)

    all_linked_entities = doc._.linkedEntities
    for entity in all_linked_entities:
        if entity != None:
            new_name = entity.get_label() + " Q" + str(entity.get_id())
            print(new_name)
            return new_name
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

# Quick fix to remove duplicate relations.
def remove_duplicates (text):
    lst = text.split(" . ")
    lst = list(dict.fromkeys(lst))
    temp = ""
    for element in lst:
        temp += element + " . "
    return temp

# Combined method containing coreference resolution, triple extraction and entity linking with graph generation.
def combined_pipeline (text, optional_coreference):
    annotated_text = text
    if(optional_coreference == True):
        annotated_text = coreference_integration(text)

    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        triple_corpus = client.annotate(annotated_text)
        # Perform entity linking on entities in the extracted triples.
        triple_corpus = entity_linking_iterative(triple_corpus)     

        graph_image = 'graph.png'
        temp = triple_list_to_string(triple_corpus)
        temp = remove_duplicates(temp)

        # Find a better tool for generating graphs
        client.generate_graphviz_graph(temp, graph_image)
        print('Graph generated: %s.' % graph_image)

# Perform triple extraction and entity linking on input text and then combine to preserve context.
def combined_joint (text, optional_coreference):
    annotated_text = text
    if(optional_coreference == True):
        annotated_text = coreference_integration(text)

    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        triple_corpus = client.annotate(annotated_text)
        # Perform entity linking on the input text.
        triple_corpus = entity_linking(annotated_text, triple_corpus)

        graph_image = 'graph.png'
        temp = triple_list_to_string(triple_corpus)
        temp = remove_duplicates(temp)

        client.generate_graphviz_graph(temp, graph_image)
        print('Graph generated: %s.' % graph_image)

# Overall architecture, specialised for the example conversation with SPN4RE
def combined_joint_spn4re (text, triples):
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        triples_linked = entity_linking_spn4re(text, triples)
        triples_linked = remove_duplicates(triples_linked)
        print(triples_linked)

        graph_image = 'graph.png'
        # Find a better tool for generating graphs
        client.generate_graphviz_graph(triples_linked, graph_image)
        print('Graph generated: %s.' % graph_image)

if __name__ == '__main__':
    data = open_corpus()
    optional_coreference = True
    
    combined_joint(data, optional_coreference)

    # Example conversation with SPN4RE applied.
    # with open('spn4re-coref.txt') as f:
            # temp = f.readlines()
            # temp = list_to_string(temp)

    # combined_joint_spn4re(data, temp)

