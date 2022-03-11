FROM pytorch/pytorch:latest

WORKDIR /code/
COPY requirements.txt requirement.txt

#Requirements
RUN pip3 install -r requirement.txt
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy_entity_linker "download_knowledge_base"
RUN apt-get update && apt-get --assume-yes install wget
RUN apt-get -y install default-jre
RUN apt-get --assume-yes install unzip

#Code Base
COPY /triple_extraction/ /code/triple_extraction/
COPY /entity_linking/ /code/entity_linking/
COPY /experiments/ /code/experiments/
COPY /coreference_resolution /code/coreference_resolution/
COPY /preprocess /code/preprocess
COPY pkg_baseline.py /code/pkg_baseline.py

#Data used to train SPN4RE relation extraction
#COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/train_new_new_v2.json /data/train.json
#COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/valid_new_new_v2.json /data/valid.json
#COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/test_new_new_v2.json /data/test.json

#Install a BERT language model for English language.
RUN apt-get --assume-yes install git
RUN git clone https://huggingface.co/bert-base-cased
RUN apt-get install git-lfs
RUN wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
RUN bash script.deb.sh && rm script.deb.sh
RUN cd bert-base-cased && git lfs pull

#get trained model
RUN pip3 install gdown && gdown --id 1SOCFaqkBcB3SFj778n3MP0QMPFri3_J9 && unzip nSetPred4RE_WebNLG_epoch_3_f1_0.3928.zip
ENV trainedmodel="/code/nSetPred4RE_WebNLG_epoch_3_f1_0.3928.model"
RUN pip3 install stanfordcorenlp #can be removed in future

#CMD ["python3", "triple_extraction/SPN4RE/predict.py"]
