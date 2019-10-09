from utils.data_loader import prepare_data_seq
from utils import config
from model.transformer import Transformer
from model.transformer_mulexpert import Transformer_experts
from model.common_layer import evaluate, count_parameters, make_infinite
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from copy import deepcopy
from tqdm import tqdm
import os
import time 
import numpy as np 
import math
from tensorboardX import SummaryWriter

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

if(config.test):
    print("Test model",config.model)
    if(config.model == "trs"):
        model = Transformer(vocab,decoder_number=program_number, model_file_path=config.save_path, is_eval=True)
    elif(config.model == "experts"):
        model = Transformer_experts(vocab,decoder_number=program_number, model_file_path=config.save_path, is_eval=True)
    if (config.USE_CUDA):
        model.cuda()
    model = model.eval()
    loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b= evaluate(model, data_loader_tst ,ty="test", max_dec_step=50)
    exit(0)

if(config.model == "trs"):
    model = Transformer(vocab,decoder_number=program_number)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
elif(config.model == "experts"):
    model = Transformer_experts(vocab,decoder_number=program_number)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n !="embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))

check_iter = 2000
try:
    if (config.USE_CUDA):
        model.cuda()
    model = model.train()
    best_ppl = 1000
    patient = 0
    writer = SummaryWriter(log_dir=config.save_path)
    weights_best = deepcopy(model.state_dict())
    data_iter = make_infinite(data_loader_tra)
    for n_iter in tqdm(range(1000000)):
        loss, ppl, bce, acc = model.train_one_batch(next(data_iter),n_iter)
        writer.add_scalars('loss', {'loss_train': loss}, n_iter)
        writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
        writer.add_scalars('bce', {'bce_train': bce}, n_iter)
        writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
        if(config.noam):
            writer.add_scalars('lr', {'learning_rata': model.optimizer._rate}, n_iter)

        if((n_iter+1)%check_iter==0):    
            model = model.eval()
            model.epoch = n_iter
            model.__id__logger = 0 
            loss_val, ppl_val, bce_val, acc_val, bleu_score_g, bleu_score_b= evaluate(model, data_loader_val ,ty="valid", max_dec_step=50)
            writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
            writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
            writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
            model = model.train()
            if (config.model == "experts" and n_iter<13000):
                continue
            if(ppl_val <= best_ppl):
                best_ppl = ppl_val
                patient = 0
                model.save_model(best_ppl,n_iter,0 ,0,bleu_score_g,bleu_score_b)
                weights_best = deepcopy(model.state_dict())
            else: 
                patient += 1
            if(patient > 2): break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

## TESTING
model.load_state_dict({ name: weights_best[name] for name in weights_best })
model.eval()
model.epoch = 100
loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b= evaluate(model, data_loader_tst ,ty="test", max_dec_step=50)

file_summary = config.save_path+"summary.txt"
with open(file_summary, 'w') as the_file:
    the_file.write("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\n")
    the_file.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\n".format("test",loss_test,ppl_test, acc_test, bleu_score_g,bleu_score_b))
    