import argparse, os, torch
import random, pickle
import numpy as np
from triple_extraction.SPN4RE.utils.data import build_data
from triple_extraction.SPN4RE.trainer.trainer import Trainer
from triple_extraction.SPN4RE.models.setpred4RE import SetPred4RE


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
        "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
        "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')

    data_arg.add_argument('--dataset_name', type=str, default="WebNLG")
    data_arg.add_argument('--train_file', type=str, default=os.getenv('traindata'))
    data_arg.add_argument('--valid_file', type=str, default=os.getenv('validdata'))
    data_arg.add_argument('--test_file', type=str, default=os.getenv('testdata'))

    data_arg.add_argument('--generated_data_directory', type=str, default=os.getenv('generated_data'))
    data_arg.add_argument('--generated_param_directory', type=str,
                          default=os.getenv('generated_data') + '/model_param/')
    data_arg.add_argument('--bert_directory', type=str, default=os.getenv('bert'))
    data_arg.add_argument("--partial", type=str2bool, default=False)

    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="Set-Prediction-Networks")
    learn_arg.add_argument('--num_generated_triples', type=int, default=int(os.getenv('num_generated_triples')))
    learn_arg.add_argument('--num_decoder_layers', type=int, default=int(os.getenv('num_decoder_layers')))
    learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    learn_arg.add_argument('--na_rel_coef', type=float, default=1)
    learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
    learn_arg.add_argument('--head_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)
    learn_arg.add_argument('--batch_size', type=int, default=8)
    learn_arg.add_argument('--max_epoch', type=int, default=os.getenv('max_epoch'))
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=0)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=100)
    evaluation_arg.add_argument('--max_span_length', type=int, default=12)  # NYT webNLG 10
    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=1)
    misc_arg.add_argument('--random_seed', type=int, default=1)

    args, unparsed = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))


    pickle.dump(args, open(os.getenv('generated_data') + "args.pickle", "wb"))
    set_seed(args.random_seed)
    data = build_data(args)
    data.show_data_summary()
    model = SetPred4RE(args, 61)

    trainer = Trainer(model, data, args)
    trainer.train_model()
