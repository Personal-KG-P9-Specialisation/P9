import json
from openie import StanfordOpenIE

def annotate(client,text):
    triple_list = list()
    for triple in client.annotate(text):
        t_list = list(triple.values())
        triple_list.append(t_list)
    return triple_list

def openie_sampling(src_file, out_file):
    f = open(src_file)
    data = json.load(f)
    f.close()

    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        for i in data:
            for j in i["messages"]:
                triple_list = annotate(client, j["utterance"])
                j["extracted_triple_openIE"] = triple_list

    f = open(out_file, "w")
    json.dump(data, f)
    f.close()

if __name__ == '__main__':
    openie_sampling('../data/random_sample/sample.json', "../data/random_sample/sample_openie_triples.json")

