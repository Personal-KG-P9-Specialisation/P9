# P9 Project - Populating Personal Knowlegde Graphs

This repository contains the experiments regarding the population of personal knowledge graphs. We have created an architecture consisting of existing components, necessary for the population of PKG. This architecture can be viewed below:

![architecture of project](https://github.com/abiram98/P9/blob/main/arch.png?raw=true)

In this architecture, the three main subtasks are namely Coreference Resolution, Triple Extraction and Entity Linking. We have created directories containing existing solutions for each of these subtasks. 
For each of the subtasks, we will conduct an experiment by annotating (Persona-chat) conversation with the output of each component. 
This output will be evaluated manually according to established criteria, which we present under each sub-task.

## Coreference Resolution

This phase concerns the extracting chains of coreference. Currently, we have experimented with:

- Stanford Core NLP Coreference Resolution

Coreference resolution is evaluated on an entire conversation at a time. 
It is deemed correct and annotated with 1, if:
1. Each coreference chain should only refer to a single entity
2. All pronouns should be present in a coreference chain

If some of the aforementioned criteria is true, it is evaluated to be partially correct and annotated with 2, 
and if none of the criteria is true, it is deemed incorrect and annotated with 0.

## Triple Extraction

Converts natural language text into triples. We have experimented with the following:

- Stanford Core NLP OpenIE
- [Joint Entity and Relation Extraction with Set Prediction Networks](https://github.com/DianboWork/SPN4RE)

Triple extraction is evaluated on each utterance manually according to the criteria:
1. They are deemed correct if (annotated with 1:
   - All annotations are extractable from the utterance
   - The triples should encapsulate all relevant facts in the utterance.
2. They are deemed partially correct if some triples with relevant personal fact are annotated (annotated with 2).
3. They are deemed contradictory if the annotated triple contains the opposite meaning of the utterance (annotated with 3).
4. They are deemed incorrect if none of the annotated triples contain relevant personal information (annotated with 0).

Besides the triple extraction approaches, we also evaluate the ground truth triples of the dataset according to the criteria.
## Entity Linking

extracts the entities from the triples and map them to entities in an open-domain knowledge graph. For this, we have tried the following:

- [Spacy Entity Linker](https://github.com/egerber/spaCy-entity-linker)

The entity linking component is divided into a recognition and a linking evaluation. The criteria is:
1. Entity recognition is deemed correct if all entities to be linked in the utterance are identified (annotated with 1)
   - They are deemed partially correct if some entities to be linked in the utterance are identified (annotated with 2)
   - They are otherwise deeemed incorrect (annotated with 0).
2. Entity linking is deemed correct if all the linked entities are linked to the correct entities (annotated with 1)
   - They are partially correct if it is only partly correct (annotated with 2)
   - They are wrong if they are linked to the incorrect entity (annotated with 0)
## Experiments on Architecture
To assess the performance of the architecture, we instantiate the architecture with two configurations:
<!---We also need to asses the architecture and therefore the combination of these existing solutions to the subtasks. Currently, we have eperimented with the following components in the architecture:--->

- Stanford Core NLP Coreference Resolution -> Stanford Core NLP OpenIE -> Spacy Entity Linker
- Stanford Core NLP Coreference Resolution -> SPN4RE -> Spacy Entity Linker

where Standford Core NLP Coreference Resolution is an optional step.
###Prerequisites
- Docker
- Docker Compose
- Nvidia GPU (for improved training speed)
- NVIDIA container toolkit
###Docker Compose
###### Random Sample
The random sample of conversations with annotations from each sub-task can be performed by first building the image, then running docker-compose on the build_pkg service:
```
docker build -t pkg:latest .
docker-compose up --remove-orphans --force-recreate -d build_pkg
```
This will create a random sample of 15 conversations annotated with each of the sub-tasks and fields for the criteria on the following format:
```
[
  {
    "id": ,
    "messages": [
      {
        "utterance": "",
        "turn": ,
        "ground_truth": ,
        "extracted_triple_openIE": [],
        "criteria_openIE": ,
        "extracted_triple_SPN4RE": [],
        "criteria_SPN4RE": ,
        "linked_entities": [
        ],
        "linked_entity_criteria": []
      }... ]
    "coreference_resolution_criteria": ,
    },...
]
```
After the sampling, we have annotated each field containing "criteria" manually using the presented criteria.
The JSON file with the manual annotation can be found in [/data/random_sample/sample_annotated.json](/data/random_sample/sample_annotated.json)
###### On own source data
To run the code on an input conversation, the format of the conversation should be a .txt where each utterance is separated by the newline character:
```
My name is X
Hello X, how are you doing?
```
To get the triples, run the following command
```
docker-compose up --remove-orphans --force-recreate -d build_pkg_from_input
```
The default conversation file is /inputs/conv.txt

To use a different file, make a volume with the conversation file with the specified path and run the following command, where CONV.txt should be replaced by the path to the conv file.
```
DATA=CONV.txt docker-compose up --remove-orphans --force-recreate -d build_pkg_from_input
```
The output triples for the conversation will be printed to the terminal, which can be accessed using:
```docker-compose logs build_pkg_from_input``` and saved in a triples_from_conv.jsonl file in the output directory.

Furthermore, the following environment variables can be changed to run the architecture on the input conversation:
- coref (Default: False. True if coref should be applied, and False if it should be omitted)
- spn (Default: True. True, if Set Prediction Network should be used as triple extractor. False, if OpenIE should be used)