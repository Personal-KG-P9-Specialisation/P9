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
#RUN mkdir -p data/generated/generated_data

#OUTPUTs
#RUN mkdir outputs
#RUN mkdir -p outputs/model
#RUN mkdir -p outputs/logs

COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/train_new_new_v2.json /data/train.json
COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/valid_new_new_v2.json /data/valid.json
COPY /triple_extraction/SPN4RE/data/WebNLG/clean_WebNLG/test_new_new_v2.json /data/test.json

ENV traindata="/data/train.json" validdata="/data/valid.json" testdata="/data/test.json" generated_data="/outputs/generated_data/" bert="/code/bert-base-cased/" modelpath="outputs/model/" logs="outputs/logs"
RUN apt-get --assume-yes install curl && apt-get install git-lfs && apt-get install wget
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN cd bert-base-cased && git lfs pull

#get trained model
RUN pip3 install gdown && gdown --id 1SOCFaqkBcB3SFj778n3MP0QMPFri3_J9 && apt-get --assume-yes install unzip && unzip nSetPred4RE_WebNLG_epoch_3_f1_0.3928.zip && mv nSetPred4RE_WebNLG_epoch_3_f1_0.3928.model code
ENV trainedmodel="/code/nSetPred4RE_WebNLG_epoch_3_f1_0.3928.model"

CMD ["python3", "triple_extraction/SPN4RE/predict.py"]
