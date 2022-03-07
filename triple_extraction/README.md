#Triple Extraction
For the triple extraction component, we provide the OpenIE implementation in openie_sampling.py file. 
This does not require a training phase. 

On the other hand, Set Prediction Network (SPN) require a training phase. 
To use, the pretrained SPN see the README in the outer directory. 

To retrain the Set prediction network for relation extraction, use the following command:
```
docker-compose up --remove-orphans --force-recreate -d train_SPN4RE
#The performance and other info can be accesed by docker-compose logs train_model 
#docker-compose up -d --remove-orphans
```
Adjustment can be made to the hyperparameters through the environment variables for the service.
The list of possible adjustment are the following:
- num_generated_triples
- num_decoder_layers
- max_epoch
