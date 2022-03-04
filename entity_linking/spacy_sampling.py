import json
import spacy

def annotate(nlp, text):
    doc = nlp(text)
    all_linked_entities = doc._.linkedEntities

    linked_entity_list = list()
    for entity in all_linked_entities:
        e_list = list()
        e_list.append(str(entity.get_span()))
        e_list.append("Q" + str(entity.get_id()))
        e_list.append(entity.get_label())
        linked_entity_list.append(e_list)

    return linked_entity_list


def entity_sampling(src_file, out_file):
    f = open(src_file)
    data = json.load(f)
    f.close()

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)

    for i in data:
        for j in i["messages"]:
            linked_entity_list = annotate(nlp, j["utterance"])
            j["linked_entities"] = linked_entity_list

    f = open(out_file, "w")
    json.dump(data, f)
    f.close()


if __name__ == '__main__':
    entity_sampling('../data/random_sample/sample.json', "../data/random_sample/sample_entities.json")
