import spacy

def pretrained_model():
    nlp = spacy.load("en_core_web_trf",disable=["tagger","parser","attribute_ruler","lemmatizer"])
    #test = """The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India and is under the jurisdiction of Ministry of Youth Affairs and Sports, Government of India.[2] The board was formed in December 1928 as a society, registered under the Tamil Nadu Societies Registration Act. It is a consortium of state cricket associations and the state associations select their representatives who in turn elect the BCCI Chief. Its headquarters are in Wankhede Stadium, Mumbai. Grant Govan was its first president and Anthony De Mello its first secretary. """
    test = "My dog's name is Elisabeth Niemeyer Laursen"
    test_nlp = nlp(test)
    for doc in test_nlp.ents:
        print(doc.text,doc.label_)
    #spacy.displacy.render(test_nlp, style="ent",jupyter=True)
if __name__ == "__main__":
    pretrained_model()
