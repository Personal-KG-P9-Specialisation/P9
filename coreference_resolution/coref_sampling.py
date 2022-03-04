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

def coref_sample(data_path, output_path, docker_service = None):
    f = open(data_path)
    data = json.load(f)
    f.close()

    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000
    if docker_service is None:
        nlp = StanfordCoreNLP('http://localhost', port=9001)
    else:
        nlp = StanfordCoreNLP('http://'+docker_service, port=9001)
    num = 0

    for i in data:
        messages = ""
        for j in i["messages"]:
            if (num % 2) == 0:
                messages += j["utterance"]
                messages += ". "
            num += 1
        coref_chains = nlp.coref(messages)
        coref_chains = convert_chains(coref_chains)

        i["coreference_chain"] = coref_chains

    nlp.close()

    f = open(output_path, "w")
    json.dump(data, f)
    f.close()

if __name__ == '__main__':
    coref_sample('../data/random_sample/sample_v2_results.json', '../data/random_sample/sample_v2_results.json')
