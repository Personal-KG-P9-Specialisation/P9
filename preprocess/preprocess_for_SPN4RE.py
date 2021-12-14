import json
from difflib import get_close_matches
def one_entity_mention(file):
    data=list()
    with open("{}.json".format(file),"r") as f:
        for l in f:
            data.append(json.loads(l))
    with open("{}_new.json".format(file),"w") as f:
        for x in data:
            new_rels = list()
            for rel in x["relationMentions"]:
                em1 = rel["em1Text"]
                em2 = rel["em2Text"]
                splits = em1.split(" ")
                if len(splits) > 1:
                    em1 = splits[-1]
                splits = em2.split(" ")
                if len(splits) > 1:
                    em2 = splits[-1]
                new_rels.append({"em1Text" : em1,"em2Text" : em2, "label":rel["label"]})
            f.write(json.dumps({"sentText":x["sentText"],"relationMentions": new_rels}))
            f.write("\n")
        f.flush()

def correct_entity_mentions(datafile:str):
    data = list()
    with open("{}.json".format(datafile), "r") as f:
        for l in f:
            data.append(json.loads(l))
    assert len(data) > 0
    with open("{}_new.json".format(datafile),"w") as f:
        for ins in data:
            new_rels = list()
            for rel in ins["relationMentions"]:
                em1 = find_closest_word(ins["sentText"], rel["em1Text"])
                em2 = find_closest_word(ins["sentText"], rel["em2Text"])
                if em1 == "" or em2 =="":
                    continue
                new_rels.append({"em1Text" : em1,"em2Text" : em2, "label":rel["label"]})
            if len(new_rels) == 0:
                continue
            f.write(json.dumps({"sentText":ins["sentText"],"relationMentions": new_rels}))
            f.write("\n")

def find_closest_word(text:str, em:str):
    text_words = text.split(" ")
    em_words = em.split(" ")
    new_em = dict()
    for e in em_words:
        try:
            closest = get_close_matches(e,text_words,1)[0]
        except IndexError:
            continue
        new_em[text.find(closest)] = closest 
    temp = sorted(new_em.keys())
    final_em = ""
    for t in temp:
        if final_em == "":
         final_em += new_em[t]
        else:
            final_em += " "
            final_em += new_em[t]
    return final_em

def filter_trpl_extr_trpls(datafile):
    data = list()
    with open("{}.json".format(datafile),"r") as f:
        for line in f:
            data.append(json.loads(line))
    assert len(data) > 0
    other_f = open("{}_other_v2.json".format(datafile), "w")
    with open("{}_new_v2.json".format(datafile),"w") as f:
        for ins in data:
            new_rels = list()
            other_rels = list()
            for rel in ins["relationMentions"]:
                if not  (rel["em1Text"] in ins["sentText"] and rel["em2Text"] in ins["sentText"]):
                    other_rels.append({"em1Text" : rel["em1Text"],"em2Text" : rel["em2Text"], "label":rel["label"]})
                else:
                    new_rels.append({"em1Text" : rel["em1Text"],"em2Text" : rel["em2Text"], "label":rel["label"]})
            if len(new_rels) == 0:
                continue
            if len(other_rels) > 0:
                other_f.write(json.dumps({"sentText":ins["sentText"],"relationMentions": other_rels}))
                other_f.write("\n")
            f.write(json.dumps({"sentText":ins["sentText"],"relationMentions": new_rels}))
            f.write("\n")
    other_f.flush()
    other_f.close()

filter_trpl_extr_trpls("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/valid_new")
filter_trpl_extr_trpls("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/test_new")
filter_trpl_extr_trpls("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/train_new")
#correct_entity_mentions("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/valid")
#correct_entity_mentions("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/test")
#correct_entity_mentions("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/train")
#one_entity_mention("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/valid")        
#one_entity_mention("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/test")
#one_entity_mention("/home/test/Github/code/SPN4RE/data/WebNLG/clean_WebNLG/train")        
