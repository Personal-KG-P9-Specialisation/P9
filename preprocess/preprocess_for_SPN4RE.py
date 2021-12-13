import json
from difflib import get_close_matches
def one_entity_mention(file):
    data=list()
    with open(file,"r") as f:
        for l in f:
            data.append(json.loads(l))
    with open("n_{}".format(file),"w") as f:
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
    with open(datafile, "r") as f:
        for l in f:
            data.append(json.loads(l))
    assert len(data) > 0
    with open("n_{}".format(datafile),"w") as f:
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




        
