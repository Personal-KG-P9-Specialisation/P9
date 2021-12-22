# P9 Project - Populating Personal Knowlegde Graphs

This repository contains the different experiments regarding the population of personal knowledge graphs. We created an architecture that would be ideal for the purpose of this project. This architecture can be viewed below:

![architecture of project](https://github.com/abiram98/P9/blob/main/arch.png?raw=true)

In this architecture, we have divided the problem into three subtasks, namely Coreference Resolution, Triple Extraction and Entity Linking. We have created folders containing some existing solutions to these subtasks. Currently, we have the following implementations.

## Coreference Resolution

This phase concerns the extracting chains of coreference. Currently, we have tried the following solutions for this phase:

- Stanford Core NLP Coreference Resolution

## Triple Extraction

Converts natural language text into triples. We have experimented with the following:

- Stanford Core NLP OpenIE
- Joint Entity and Relation Extraction with Set Prediction Networks

## Entity Linking

Should extract the entities from the triples and map them to entities in an existing general pupose knowledge graph. For this, we have tried the following:

- Spacy Entity Linker

## Experiments on Architecture

We also need to asses the architecture and therefore the combination of these existing solutions to the subtasks. Currently, we have eperimented with the following components in the architecture:

- Stanford Core NLP Coreference Resolution -> Stanford Core NLP OpenIE -> Spacy Entity Linker
