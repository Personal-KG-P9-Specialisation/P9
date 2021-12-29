import json

datafolder = "../data/ConvAI2"
class Message:
    def __init__(self, utterance:str, turn:int):
        self.utterance = utterance
        self.turn = turn
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

