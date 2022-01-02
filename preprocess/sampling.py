import json
import numpy as np

#Class for a single utterance
class Message:
    def __init__(self, utterance:str, turn:int):
        self.utterance = utterance
        self.turn = turn
        self.ground_truth = list()
        self.extracted_triple_openIE = list()
        self.criteria_openIE = [0,0]
        self.extracted_triple_SPN4RE = list()
        self.criteria_SPN4RE = [0,0]
        self.linked_entities = list()
        self.linked_entity_criteria=[0,0]
    def add_gt(self,gt:str):
        self.ground_truth.append(gt)
    def __repr__(self):
        return repr(vars(self))

class Dialogue:
    def __init__(self, id:int) -> None:
        self.id = id
        self.messages = list()
        self.coreference_chain = list()
        self.coreference_resolution_criteria = [0,0]
    def add_msg(self,message:Message):
        self.messages.append(message)

#Returns amount of dialogues in pesona-chat style txt files
def count_dialogues(file):
    dial_count = 0
    last =""
    with open(file,"r") as f:
        data = f.readlines()
    for l in data:
        if l.startswith("your persona"):
            last = "your persona"
            continue
        elif l.startswith("partner's persona"):
            if last == "your persona":
                dial_count += 1
            last = "partner's persona"
            continue        
    print("Number of Dialogues in {} is {}".format(file,dial_count))

#reformats personastyle rxt into objects
def create_json_data(file):
    dialog_id = 0
    last =""
    with open(file,"r") as f:
        data = f.readlines()
    dialogues = list()
    d = None
    for l in data:
        if l.startswith("your persona"):
            last = "your persona"
            continue
        elif l.startswith("partner's persona"):
            if last == "your persona":
                dialog_id += 1
            last = "partner's persona"
            continue
        else:
            txt_spl = l.split("\t")
            if(int(txt_spl[0]) == 1 and d is None):
                d = Dialogue(dialog_id)
            elif(int(txt_spl[0]) == 1 and d is not None):
                dialogues.append(d)
                d = Dialogue(dialog_id)
            if txt_spl[1][len(txt_spl[1])-1] == "\n":
                msg = Message(txt_spl[1][:-1],int(txt_spl[0]))
            else:
                msg = Message(txt_spl[1],int(txt_spl[0]))
            if len(txt_spl) > 2:
                if(txt_spl[2][len(txt_spl[2])-1] == "\n"):
                    msg.add_gt(txt_spl[2][:-1])
                else:
                    msg.add_gt(txt_spl[2])
            d.add_msg(msg)
        
    #print(json.dumps(dialogues,default=lambda a : getattr(a,'__dict__', str(a))))
    #json.dump(dialogues,open("sample.json","w"),default=lambda a : getattr(a,'__dict__', str(a)))
    return dialogues

#967 dialogues in test set
def random_sample(dials, N):
    rand_idx = list(np.random.permutation(np.arange(1,967))[:N])
    print(rand_idx)
    sample_dials = [x for x in dials if x.id in rand_idx]
    json.dump(sample_dials,open("sample.json","w"),default=lambda a : getattr(a,'__dict__', str(a)))

if __name__ == "__main__":
    count_dialogues("../data/ConvAI2/test_both_original_final.txt")
    data = create_json_data("../data/ConvAI2/test_both_original_final.txt")    
    random_sample(data,40)
#ids for 40 first
#[298, 157, 864, 326, 300, 433, 74, 599, 664, 512, 705, 7, 37, 624, 261, 232, 866, 335, 941, 875, 748, 717, 468, 136, 474, 101, 841, 773, 965, 250, 775, 683, 199, 649, 370, 613, 884, 742, 359, 602]