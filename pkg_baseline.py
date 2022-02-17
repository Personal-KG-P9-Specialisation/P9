from openie import StanfordOpenIE
import spacy
import json
from stanfordcorenlp import StanfordCoreNLP
import re
from experiments.architecture import entity_linking_spn4re



if __name__ == "__main__":
    data = entity_linking_spn4re("My name is X","[('My name', 'is', 'X')]")
    print(data)
    pass
