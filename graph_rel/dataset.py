from torch.utils.data import Dataset
import spacy, json
class DS(Dataset):
    def __init__(self, data):
        super(DS,self).__init__()

        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

tr = json.loads(open("nyttest.json","r").read())
NUM_REL = dict()
MXL = 0

for d in tr:
    sent = d["sentText"]
    MXL = max(len(sent)+4, MXL)
    
    rels = d["relationMentions"]
    
    for rel in rels:
        
        if not rel in NUM_REL:
            NUM_REL[rel] = 1

print(NUM_REL)
            
NUM_REL = len(NUM_REL) + 1 # 0 for NA
print('Num of relation: %d' % (NUM_REL))
print('Max length: %d' % (MXL))
