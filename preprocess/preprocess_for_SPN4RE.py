import json

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

        
