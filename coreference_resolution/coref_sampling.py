import json
from stanfordcorenlp import StanfordCoreNLP

def convert_chains (coref_chains):
    new_coref_chains = list()
    for chain in coref_chains:
        corefs = list()
        for ref in chain:
            corefs.append(ref[3])
        new_coref_chains.append(corefs)
    return new_coref_chains

if __name__ == '__main__':
    f = open('../data/random_sample/sample.json')
    data = json.load(f)
    f.close()

    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001)

    for i in data:
        messages = ""
        for j in i["messages"]:
            messages += j["utterance"]
            messages += ". "
        coref_chains = nlp.coref(messages)
        coref_chains = convert_chains(coref_chains)

        i["coreference_chain"] = coref_chains

    nlp.close()

    f = open("../data/random_sample/sample_coref.json", "w")
    json.dump(data, f)
    f.close()
