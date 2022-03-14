#FROM python:3.8
FROM openjdk:8

WORKDIR /code/
RUN apt-get update && apt-get --assume-yes install wget
RUN wget https://nlp.stanford.edu/software/stanford-corenlp-4.1.0.zip
RUN apt-get --assume-yes install unzip
RUN unzip stanford-corenlp-4.1.0.zip
RUN export CLASSPATH=$CLASSPATH:/stanford-corenlp-4.1.0:
#RUN cd stanford-corenlp-4.4.0 && java -mx4g -cp "stanford-corenlp-4.4.0/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000
