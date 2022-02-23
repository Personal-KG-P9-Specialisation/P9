from openie import StanfordOpenIE
import spacy
import json
from stanfordcorenlp import StanfordCoreNLP
import re
from experiments.architecture import entity_linking
from triple_extraction.SPN4RE.predict import load_model_tok_data


if __name__ == "__main__":
    load_model_tok_data(os.getenv('trainedmodel'))
    triple = {'subject':'My', 'object':'Michael Jordan' }
    triples = [triple]
    data = entity_linking("My name is Michael Jordan",triples)
    print(data)
    pass
