import os
import sys

from openie import StanfordOpenIE
import spacy
import json
import re
from experiments.architecture import entity_linking, openie_extract_triples, coreference_integration
from triple_extraction.SPN4RE.predict import predict_utterance, predict_data
from preprocess.sampling import create_json_data, random_sample, create_conversation
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

def coref_on_conv(conv):
    text = ''
    for utt in conv.messages:
        text = text + utt.utterance+'\n'
    text = coreference_integration(text)
    utts = text.split('\n');count=0
    for utt in conv.messages:
        utt.utterance = utts[count]
        count +=1
    return conv


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

def triple_equal(trpl1, trpl2):
    if trpl1['subject'] == trpl2['subject'] and trpl1['object']== trpl2['object'] and trpl1['relation']== trpl2['relation']:
        return True
    else:
        return False

def is_triple_in_lst(trpl, trpl_lst):
    for x in trpl_lst:
        if triple_equal(x,trpl):
            return True
    return False

def triple_integrate(trpls, new_trpls):
    """if len(trpls)==0:
        return new_trpls"""
    for x in new_trpls:
        if not is_triple_in_lst(x,trpls):
            trpls.append(x)
    return trpls

def post_process(triples):
    f = open('/code/outputs/triples_from_conv.jsonl','w')
    for t in triples:
        f.write(json.dumps(t))
        f.write('\n')
    f.close()
    print(triples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, nargs='?',
                        help='type \'sample\' for sample creation and \'architecture\' for annotation of text')
    parser.add_argument('--data', type=str, nargs='?', default=os.getenv('DATA'))
    parser.add_argument('--coref', type=str2bool, nargs='?', const=True, default=os.getenv('coref'),
                        help='coref can be added by passing true to this flag')
    parser.add_argument('--SPN4RE', type=str2bool, nargs='?', const=True, default=os.getenv('spn'),
                        help='SPN4RE instead of OPENIE triple extraction')

    args = parser.parse_args()

    if args.type == 'sample':
        create_sample()
    elif args.type == 'conv':
        conv = create_conversation(args.data)
        triples = list()
        if args.coref == True:
            conv = coref_on_conv(conv)
            if args.SPN4RE:
                for utt in conv.messages:
                    triples = triple_integrate(triples,triple_and_ent_link(utt.utterance))
                post_process(triples)
            else:
                for utt in conv.messages:
                    triples = triple_integrate(triples,triple_and_ent_link(utt.utterance, triple='openIE'))
                post_process(triples)
        else:
            if args.SPN4RE:
                for utt in conv.messages:
                    triples = triple_integrate(triples,triple_and_ent_link(utt.utterance))
                post_process(triples)
            else:
                for utt in conv.messages:
                    triples = triple_integrate(triples,triple_and_ent_link(utt.utterance, triple='openIE'))
                post_process(triples)