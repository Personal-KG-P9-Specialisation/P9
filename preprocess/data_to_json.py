import json,ast, re

generated_data = "generated_data"
"generated_test.json"
datafolder = "../data/ConvAI2"
test_data_file = "test_both_original_final.txt"
train_data_file = "train_both_original_final.txt"
val_data_file = "valid_both_original_final.txt"
output_folder = "../data/SPN4RE_data"
#convert triple to json format usable by SPN4RE
def str_tpl_proc(tpl):
    tmp = {}
    tmp["em1Text"] = tpl[0]
    tmp["em2Text"] = tpl[2]
    tmp["label"] = tpl[1]
    return tmp

def to_json(folder, data_file, out_folder, out_file):
    f = open("{}/{}".format(folder,data_file), "r")
    data = list()
    for line in f:
        if re.match("partner",line) or re.match("your persona",line):
            continue
        text_dict = text_to_json(line)
        if not text_dict is None:
            data.append(json.dumps(text_dict)+"\n")
    f.close()
    with open("{}/{}".format(out_folder,out_file),"w") as f:
        for line in data:
            f.write(line)
        f.flush()
#return object if triples are present 
def text_to_json(text):
    splits = text.split("\t")
    splits = splits[1:]
    text_dict = {}
    text_dict["sentText"] = splits[0].replace("\n","")
    text_dict["relationMentions"] =[str_tpl_proc(ast.literal_eval(x)) for x in splits[1:]]
    if len(text_dict["relationMentions"]) == 0:
        return None
    return text_dict
def generate_data():
    to_json(datafolder,test_data_file,output_folder,"test.json")
    to_json(datafolder,train_data_file,output_folder,"train.json")
    to_json(datafolder,val_data_file,output_folder,"valid.json")
generate_data()
"""f = open("{}/{}".format("../data/ConvAI2","test_both_original_final.txt"), "r")
for line in f:
    if re.match("partner",line) or re.match("your persona",line):
        continue
    text_dict = text_to_json(line)
    if not text_dict is None:
        print(json.dumps(text_dict))
        break
f.close()"""
"""text = "	i am great . i just got back from the club .	['i', 'like_activity', 'dancing']	['i', 'live_in_general', 'apartment']"
splits = text.split("\t")
splits = splits[1:]
text_dict = {}
text_dict["sentText"] = splits[0]
text_dict["relationMentions"] =[str_tpl_proc(ast.literal_eval(x)) for x in splits[1:]]
print(splits)
print(json.dumps(text_dict))"""