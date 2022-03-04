# P9 Project - Populating Personal Knowlegde Graphs

This repository contains the experiments regarding the population of personal knowledge graphs. We created an architecture consisting of existing components, necessary for the population of PKG. This architecture can be viewed below:

![architecture of project](https://github.com/abiram98/P9/blob/main/arch.png?raw=true)

In this architecture, the three main subtasks are namely Coreference Resolution, Triple Extraction and Entity Linking. We have created directories containing existing solutions for each of these subtasks. 
For each of the subtasks, we will conduct an experiment by annotating (Persona-chat) conversation with the output of each 

## Coreference Resolution

This phase concerns the extracting chains of coreference. Currently, we have experiemnted with:

- Stanford Core NLP Coreference Resolution

## Triple Extraction

Converts natural language text into triples. We have experimented with the following:

- Stanford Core NLP OpenIE
- [Joint Entity and Relation Extraction with Set Prediction Networks](https://github.com/DianboWork/SPN4RE)

## Entity Linking

extracts the entities from the triples and map them to entities in an open-domain knowledge graph. For this, we have tried the following:

- [Spacy Entity Linker](https://github.com/egerber/spaCy-entity-linker)

## Experiments on Architecture
To assess the performance of the architecture, we instantiate the architecture with two configurations:
<!---We also need to asses the architecture and therefore the combination of these existing solutions to the subtasks. Currently, we have eperimented with the following components in the architecture:--->

- Stanford Core NLP Coreference Resolution -> Stanford Core NLP OpenIE -> Spacy Entity Linker
- Stanford Core NLP Coreference Resolution -> SPN4RE -> Spacy Entity Linker

For the actual experiment, we evaluate each comp
###Prerequisites
- Docker
- Docker Compose
- Nvidia GPU (for improved training speed)
- NVIDIA container toolkit
###Docker
A docker image can first be build by cd'ing into the working directory and by using:
```
docker build -t pkg:latest .
```
To retrain the Set prediction network for relation extraction, use the following command:
```
docker-compose up train_model
#The performance and other info can be accesed by docker-compose logs train_model 
#docker-compose up -d --remove-orphans
```

To run the code, docker can be used.
```
docker build -t test:test . #building the image
#docker run -it --rm --name devtest --mount type=bind,src="$(pwd)"/outputs,target=/code/outputs --mount type=bind,src="$(pwd)"/outputs,target=/code/data test:test python3 train.py
docker-compose up -d --remove-orphans
docker-compose down --remove-orphans # command for removing the docker instances.
```