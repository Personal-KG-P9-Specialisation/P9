import json
from openie import StanfordOpenIE

def annotate(client,text):
    triple_list = list()
    for triple in client.annotate(text):
        t_list = list(triple.values())
        triple_list.append(t_list)
    return triple_list

if __name__ == '__main__':
    f = open('sampling.json')
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

    f = open("sampling_triples.json", "w")
    json.dump(data, f)
    f.close()
