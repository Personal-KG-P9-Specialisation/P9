import argparse, os, torch
import random
import numpy as np
from models.setpred4RE import SetPred4RE
from utils.data import build_data

def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
    "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
    "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg
def str2bool(v):
    return v.lower() in ('true')

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

def prepare_data():
    st = "hello"

"""
    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False).cuda() for k, v in t.items()} for t in targets]
        else:
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in targets]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info
"""
def tokenise():
    d = "hello, my name is X"

    pass
def from_main():
    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')
    
    # data_arg.add_argument('--dataset_name', type=str, default="NYT-exact")
    # data_arg.add_argument('--train_file', type=str, default="./data/NYT/exact_data/train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/NYT/exact_data/valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/NYT/exact_data/test.json")
    
    # data_arg.add_argument('--dataset_name', type=str, default="NYT-partial")
    # data_arg.add_argument('--train_file', type=str, default="./data/NYT/casrel_data/new_train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/NYT/casrel_data/new_valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/NYT/casrel_data/new_test.json")
    
    
    data_arg.add_argument('--dataset_name', type=str, default="WebNLG")
    data_arg.add_argument('--train_file', type=str, default="./data/WebNLG/clean_WebNLG/train_new_new_v2.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/WebNLG/clean_WebNLG/valid_new_new_v2.json")
    data_arg.add_argument('--test_file', type=str, default="./data/WebNLG/clean_WebNLG/test_new_new_v2.json")

    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--bert_directory', type=str, default="./bert_base_cased/")
    data_arg.add_argument("--partial", type=str2bool, default=False)
    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="Set-Prediction-Networks")
    learn_arg.add_argument('--num_generated_triples', type=int, default=10)
    learn_arg.add_argument('--num_decoder_layers', type=int, default=3)
    learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    learn_arg.add_argument('--na_rel_coef', type=float, default=1)
    learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
    learn_arg.add_argument('--head_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)
    learn_arg.add_argument('--batch_size', type=int, default=8)
    learn_arg.add_argument('--max_epoch', type=int, default=50)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=0)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=100)
    evaluation_arg.add_argument('--max_span_length', type=int, default=12) #NYT webNLG 10
    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=1)
    misc_arg.add_argument('--random_seed', type=int, default=1)

    args, unparsed = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    for arg in vars(args):
            print(arg, ":",  getattr(args, arg))
    set_seed(args.random_seed)
    return args
args = from_main()
try:
    from transformers import BertTokenizer
except:
    from pytorch_transformers import BertTokenizer

def predict_triples(utterance, model, tokenizer, data):
    token_sent = [tokenizer.cls_token] + tokenizer.tokenize(remove_accents(utterance)) + [tokenizer.sep_token]
    sent_ids = tokenizer.convert_tokens_to_ids(token_sent)
    max_sent_len = len(sent_ids)
    input_ids = torch.zeros((1, max_sent_len), requires_grad=False).long()
    attention_mask = torch.zeros((1, max_sent_len), requires_grad=False, dtype=torch.float32)
    input_ids[0, :max_sent_len] = torch.LongTensor(sent_ids)
    attention_mask[0, :max_sent_len] = torch.FloatTensor([1] * max_sent_len)
    info ={"seq_len":[max_sent_len],"sent_idx":[0]}
    output = model.gen_triples(input_ids, attention_mask, info)
    relation = output[0][0][0]
    relation = data.relational_alphabet.get_instance(relation)


    pass
tokenizer = BertTokenizer.from_pretrained("BERT DIR", do_lower_case=False)
data = build_data(args)
model = SetPred4RE(args, data.relational_alphabet.size())

