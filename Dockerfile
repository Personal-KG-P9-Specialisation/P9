#FROM python:3.8
FROM pytorch/pytorch:latest

WORKDIR /code/
COPY requirements.txt requirement.txt

#RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install -r requirement.txt
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy_entity_linker "download_knowledge_base"

COPY /triple_extraction/ /code/triple_extraction/
COPY /entity_linking/ /code/entity_linking/
COPY /experiments/ /code/experiments/
COPY pkg_baseline.py /code/pkg_baseline.py
RUN apt-get update && apt-get --assume-yes install git
RUN git clone https://huggingface.co/bert-base-cased

##TTAD dataset
RUN mkdir data
RUN mkdir -p data/generated
RUN mkdir -p data/generated/generated_data

COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/train_new_new_v2.json /data/train.json
COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/valid_new_new_v2.json /data/valid.json
COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/test_new_new_v2.json /data/test.json

ENV traindata="/data/train.json" validdata="/data/valid.json" testdata="/data/test.json" generated_data="/data/generated/generated_data/" bert="./bert_base_cased/" modelpath="" logs=""

CMD ["python3", "triple_extraction/SPN4RE/predict.py"]
