import spacy  # version 3.0.6'

nlp = spacy.load("en_core_web_md")

nlp.add_pipe("entityLinker", last=True)

# first message in the example from the report
doc = nlp("I have neighbors. My neighbors dog won\'t stop barking at me. Ugh!")

# returns all linked entities
all_linked_entities = doc._.linkedEntities

# iterates over linked entities and prints them
for entity in all_linked_entities:
    entity.pretty_print()