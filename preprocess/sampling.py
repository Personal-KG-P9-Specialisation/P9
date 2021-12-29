import json

datafolder = "../data/ConvAI2"
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
    def __repr__(self):
        return repr(vars(self))

class Dialogue:
    def __init__(self, id:int, messages) -> None:
        self.id = id
        self.messages = messages
        self.coreference_chain = list()
        self.coreference_resolution_criteria = [0,0]

def read_data(file):
    dialog_id = 0
    last =""
    with open(file,"r") as f:
        data = f.readlines()
    for l in data:
        if l.startswith("your persona"):
            last = "your persona"
            continue
        if l.startswith("partner's persona"):
            if last == "your persona":
                dialog_id += 1
            last = "partner's persona"
            continue
        
    print(dialog_id)
read_data("../data/ConvAI2/valid_both_original_final.txt")
"""lst = list()
lst.append(Message("hello",1))
lst.append(Message("hi",2))
d = Dialogue(1,lst)
print(json.dumps(d,default=lambda a : getattr(a,'__dict__', str(a))))"""