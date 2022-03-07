import os
import sys

from openie import StanfordOpenIE
import spacy
import json
import re
from experiments.architecture import entity_linking, openie_extract_triples, coreference_integration
from triple_extraction.SPN4RE.predict import predict_utterance, predict_data
from preprocess.sampling import create_json_data, random_sample
from coreference_resolution.coref_sampling import coref_sample
from entity_linking.spacy_sampling import entity_sampling
from triple_extraction.openie_sampling import openie_sampling
import argparse


def triple_and_ent_link(utterance, triple='spn'):
    if triple == 'spn':
        triples = predict_utterance(utterance)
    else:
        triples = openie_extract_triples(utterance)
    return entity_linking(utterance, triples)


def triple_and_coref_ent_link(utterance, triple='spn'):
    utterance = coreference_integration(utterance)
    return triple_and_ent_link(utterance, triple=triple)


def create_sample():
    data_source = os.getenv('datafile')
    output_file = os.getenv('samplefile')

    dialogues = create_json_data(data_source)
    random_sample(dialogues, 15, output_file)
    print('Random sampling completed\n')
    coref_sample(output_file, output_file, docker_service='run_coref')
    print('Coreferencing completed\n')
    entity_sampling(output_file, output_file)
    print('Entity Linking completed')
    openie_sampling(output_file, output_file)
    print('OpenIE triple extraction completed')
    predict_data(output_file, output_file)
    print('SPN4RE triple extraction completed')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, nargs='?',
                        help='type \'sample\' for sample creation and \'architecture\' for annotation of text')
    parser.add_argument('--data', type=str, nargs='?')
    parser.add_argument('--coref', type=str2bool, nargs='?',const=True,default=False,
                        help='coref can be added by passing true to this flag')
    parser.add_argument('--SPN4RE', type=str2bool, nargs='?', const=True, default=True,
                        help='SPN4RE instead of OPENIE triple extraction')

    args = parser.parse_args()
    for arg in vars(args):
            print(arg, ":",  getattr(args, arg))
    if args.type == 'sample':
        create_sample()
    elif args.type == 'architecture':
        pass
