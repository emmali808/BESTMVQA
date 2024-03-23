"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
from attention import BiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading
from simple_cnn import SimpleCNN, SimpleCNN32
from learner import MAML
from auto_encoder import Auto_Encoder_Model
import os
# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb, ae_v_emb):
        super(BAN_Model, self).__init__()
        self.args = args
        self.dataset = dataset
        self.op = args.op
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()
        if args.maml:
            # init multiple maml models
            if len(self.args.maml_nums) > 1:
                self.maml_v_emb = nn.ModuleList(model for model in maml_v_emb)
            else:
                self.maml_v_emb = maml_v_emb
            ## later by hxj
            if 'SLAKE' in self.args.VQA_dir:
                self.convert_maml = nn.Linear(64, 32)

        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, args.feat_dim)
    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml: # get maml feature
            # compute multiple maml embeddings and concatenate them
            if len(self.args.maml_nums) > 1:
                maml_v_emb = self.maml_v_emb[0](v[0]).unsqueeze(1)
                for j in range(1, len(self.maml_v_emb)):
                    maml_v_emb_temp = self.maml_v_emb[j](v[0]).unsqueeze(1)
                    maml_v_emb = torch.cat((maml_v_emb, maml_v_emb_temp), 2)
            else:
                maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
                ## alter by hxj
                if 'SLAKE' in self.args.VQA_dir:
                    maml_v_emb =  self.convert_maml(maml_v_emb)
            v_emb = maml_v_emb
        if self.args.autoencoder: # get dae feature
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder: # concatenate maml feature with dae feature
            # print("#####",maml_v_emb.shape,  ae_v_emb.shape)
            # exit()
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        # get lextual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        if self.args.autoencoder:
                return q_emb.sum(1), decoder
        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Create SAN model
class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb):
        super(SAN_Model, self).__init__()
        self.args = args
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        if args.maml:
            # init multiple maml models
            if len(self.args.maml_nums) > 1:
                self.maml_v_emb = nn.ModuleList(model for model in maml_v_emb)
            else:
                self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, args.feat_dim)
    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml: # get maml feature
            # compute multiple maml embeddings and concatenate them
            if len(self.args.maml_nums) > 1:
                maml_v_emb = self.maml_v_emb[0](v[0]).unsqueeze(1)
                for j in range(1, len(self.maml_v_emb)):
                    maml_v_emb_temp = self.maml_v_emb[j](v[0]).unsqueeze(1)
                    maml_v_emb = torch.cat((maml_v_emb, maml_v_emb_temp), 2)
            else:
                maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder: # get dae feature
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder: # concatenate maml feature with dae feature
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        # get textual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state
        # Attention
        att = self.v_att(v_emb, q_emb)
        if self.args.autoencoder:
            return att, decoder
        return att

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Build BAN model
def build_BAN(dataset, args, priotize_using_counter=False):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0,  args.rnn)
    v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    # build and load pre-trained MAML model(s)
    if args.maml:
        # load multiple pre-trained maml models(s)
        if  'SLAKE' in args.VQA_dir or 'Med-2019' in args.VQA_dir  or 'OVQA' in args.VQA_dir:
            weight_path = os.path.join(args.VQA_dir, args.maml_model_path)
            maml_v_emb = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)
            # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
        else:
            if len(args.maml_nums) > 1:
                maml_v_emb = []
                for model_t in args.maml_nums:
                    ## alter by hxj
                    weight_path = args.VQA_dir + '/maml/' + 't%s_'%(model_t) + args.maml_model_path
                    
                    print('load initial weights MAML from: %s' % (weight_path))
                    # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
                    maml_v_emb_temp = MAML(args.VQA_dir)
                    maml_v_emb_temp.load_state_dict(torch.load(weight_path))
                    maml_v_emb.append(maml_v_emb_temp)
            else:
                weight_path = args.VQA_dir + '/maml/' + 't%s_' % (args.maml_nums[0]) + args.maml_model_path
                print('load initial weights MAML from: %s' % (weight_path))
                # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
                maml_v_emb = MAML(args.VQA_dir)
                maml_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.VQA_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # Optional module: counter for BAN
    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    if use_counter or priotize_using_counter:
        counter = Counter(objects)
    else:
        counter = None
    # init BAN residual network
    b_net = []
    q_prj = []
    c_prj = []
    for i in range(args.gamma):
        b_net.append(BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         ae_v_emb)
    elif args.maml:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         None)
    elif args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None,
                         ae_v_emb)
    return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None, None)

# Build SAN model
def build_SAN(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0, args.rnn)
    v_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)
    # build and load pre-trained MAML model(s)
    if args.maml:
        # load multiple pre-trained maml models(s)
        if len(args.maml_nums) > 1:
            maml_v_emb = []
            for model_t in args.maml_nums:
                weight_path = args.VQA_dir + '/maml/' + 't%s_'%(model_t) + args.maml_model_path
                print('load initial weights MAML from: %s' % (weight_path))
                # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
                maml_v_emb_temp = MAML(args.VQA_dir)
                maml_v_emb_temp.load_state_dict(torch.load(weight_path))
                maml_v_emb.append(maml_v_emb_temp)
        else:
            weight_path = args.VQA_dir + '/maml/' + 't%s_' % (args.maml_nums[0]) + args.maml_model_path
            print('load initial weights MAML from: %s' % (weight_path))
            maml_v_emb = MAML(args.VQA_dir)
            maml_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.VQA_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb)
    elif args.maml:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, None)
    elif args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, ae_v_emb)
    return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, None)