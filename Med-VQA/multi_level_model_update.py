# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         model
# Description:  BAN model [Bilinear attention + Bilinear residual network]
# Author:       Boliu.Kelvin
# Date:         2020/4/7
#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from connect import FCNet
from connect import BCNet
from counting import Counter
from utils import tfidf_loading
from maml import SimpleCNN
from auto_encoder import Auto_Encoder_Model
from torch.nn.utils.weight_norm import weight_norm
from classify_question import typeAttention
# from vilmedic import AutoModel
from classify_question import QuestionAttention
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
from tools.create_dictionary import Dictionary
# from multi_task import Multi_task_Model
# import paddle
# import paddle.nn as nn
# import math
def linear(in_dim, out_dim, bias=True):

    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin
# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):  # v:32,1,128; q:32,12,1024
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits

class BiResNet(nn.Module):
    def __init__(self,args,dataset,priotize_using_counter=False,use_report=False):
        super(BiResNet,self).__init__()
        # Optional module: counter
        use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10  # minimum number of boxes
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None
        # # init Bilinear residual network
        b_net = []   # bilinear connect :  (XTU)T A (YTV)
        q_prj = []   # output of bilinear connect + original question-> new question    Wq_ +q
        c_prj = []
        for i in range(args.glimpse):
            if use_report:
                b_net.append(BCNet(383, args.hid_dim, args.hid_dim, None, k=1))
            else:
                b_net.append(BCNet(dataset.v_dim, args.hid_dim, args.hid_dim, None, k=1))
            
            q_prj.append(FCNet([args.hid_dim, args.hid_dim], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, args.hid_dim], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.args = args

    def forward(self, v_emb, q_emb,att_p):
        b_emb = [0] * self.args.glimpse
        for g in range(self.args.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:]) # b x l x h
            # atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb.sum(1)


def seperate(v,q,a,att,answer_target,args):     #q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b

    indexs_open = []
    indexs_close = []
    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
    if 'OVQA' in args.data_dir:# 删掉，待定
        close_num = 321
        all_num = 1102
    elif ('RAD' in args.data_dir) or ('FREE' in args.data_dir):
        close_num=56
        all_num=487
    elif 'SLAKE' in args.data_dir:
        '''
                train_test
                close_num = 35
                all_num = 254
                '''
        close_num = 36
        all_num = 257
    elif 'PATH' in args.data_dir:
        '''
        close_num = 2
        all_num = 4122
        '''
        close_num = 2
        all_num = 4903

    elif 'Med-2019' in args.data_dir:
        close_num = 2
        all_num = 1607    

    if att != None:
        return v[indexs_open, :, :], v[indexs_close, :, :], q[indexs_open, :, :], q[indexs_close, :, :], a[indexs_open,close_num:all_num], a[indexs_close,:close_num], att[indexs_open,:], att[indexs_close,:]
    else:
        return v[indexs_open, :, :], v[indexs_close, :, :], q[indexs_open, :, :], q[indexs_close, :, :], a[indexs_open,close_num:all_num], a[indexs_close,:close_num], None, None


def seperate_test(v, q, a, att, answer_target, args,qid):  # q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []
    for i in range(len(answer_target)):
        if answer_target[i] == 0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
    if 'OVQA' in args.data_dir:# 1102 781 321 num of all open closed
        close_num = 321
        all_num = 1102
    elif ('RAD' in args.data_dir) or ('FREE' in args.data_dir):
        close_num = 56
        all_num = 487
    elif 'SLAKE' in args.data_dir:
        '''
        train_test
        close_num = 35
        all_num = 254
        '''
        close_num = 36
        all_num = 257
    elif 'PATH' in args.data_dir:
        '''
                close_num = 2
                all_num = 4122
                '''
        close_num = 2
        all_num = 4903
    elif 'Med-2019' in args.data_dir:
        close_num = 2
        all_num = 1607   

    print("#####", v.shape, q.shape, a.shape)
    print("#####", type(qid))
    print("#####", indexs_open, indexs_close)
    print("#####", type(indexs_open), type(indexs_close))
    aa = v[indexs_close, :, :]
    db = q[indexs_close, :, :]
    dfd = a[indexs_close, : close_num]
    dfsf = qid[indexs_close]


    # print("#####", type(v)
        
    if att != None:
        return v[indexs_open, :, :], v[indexs_close, :, :], q[indexs_open, :, :], q[indexs_close, :, :], a[indexs_open, close_num:all_num], a[indexs_close,:close_num], att[indexs_open,:], att[indexs_close,:],qid[indexs_open], qid[indexs_close]
    else:
        # if len(indexs_open) == 0:
        #     qid_open = None
        # else:
        #     qid_open = qid[indexs_open]
        # if len(indexs_close) == 0:
        #     qid_close = None
        # else:
        #     qid_close = qid[indexs_close]  
        if len(indexs_open) == 0:  
            return None, v[indexs_close, :, :], None, q[indexs_close, :, :], None, a[indexs_close, : close_num], None, None, None, qid[indexs_close]
        if len(indexs_close) == 0:
            return v[indexs_open, :, :], None, q[indexs_open, :, :],None, a[indexs_open,close_num:all_num], None, None, None,qid[indexs_open],None
             
        return v[indexs_open, :, :], v[indexs_close, :, :], q[indexs_open, :, :], q[indexs_close, :, :], a[indexs_open,close_num:all_num], a[indexs_close,:close_num], None, None,qid[indexs_open], qid[indexs_close]


def seperate_report(report_emb,answer_target):     #q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []
    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
    return report_emb[indexs_open, :, :], report_emb[indexs_close, :, :]



# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset,args):
        super(BAN_Model, self).__init__()

        self.args = args
        self.dataset=dataset
        # init word embedding module, question embedding module, biAttention network, bi_residual network, and classifier
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.cat)
        # print("########", self.w_emb.cat)
        self.q_emb = QuestionEmbedding(600 if args.cat else 300, args.hid_dim, 1, False, .0, args.rnn)

        # for close att+ resnet + classify
        self.close_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        self.close_resnet = BiResNet(args, dataset)
        self.close_classifier = SimpleClassifier(args.hid_dim, args.hid_dim * 2, dataset.num_close_candidates, args)



        # for open_att + resnet + classify
        self.open_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        self.open_resnet = BiResNet(args, dataset)
        self.open_classifier = SimpleClassifier(args.hid_dim, args.hid_dim * 2, dataset.num_open_candidates, args)

        # for image and question classify
        if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset)or ('FREE' in self.args.dataset):
            if args.image_classify:
                # self.image_classifier = SimpleClassifier(832, 832 * 2, 3, args)
                if args.gloria_global and args.att_visual:
                    # self.image_classifier = SimpleClassifier(16896, 2048, len(dataset.organ_list), args) #words_num=20 16896=(words_num+2)*768
                    # self.image_classifier = SimpleClassifier(7680, 2048, len(dataset.organ_list), args)#words_num=8
                    # self.image_classifier = SimpleClassifier(8448, 2048, len(dataset.organ_list), args)  # words_num=9
                    # self.image_classifier = SimpleClassifier(13824, 2048, len(dataset.organ_list), args)  # words_num=1
                    self.image_classifier = SimpleClassifier(args.v_dim, 2048, len(dataset.organ_list),
                                                             args)  # words_num=20
                elif args.gloria_global and not args.att_visual:
                    self.image_classifier = SimpleClassifier(832, 2048, len(dataset.organ_list), args)
                elif not args.gloria_global and  args.att_visual:
                    self.image_classifier = SimpleClassifier(16128, 2048, len(dataset.organ_list), args)
            if args.question_classify:
                self.q_final = QuestionAttention(1024)
                self.question_classifier = SimpleClassifier(12 * args.hid_dim, 2048, len(dataset.question_type_list),
                                                            args)
        

        # type attention: b * 1024
        self.typeatt = typeAttention(dataset.dictionary.ntoken,args.data_dir+'/glove6b_init_300d.npy')

        # #Gloria model
        # if args.gloria_global or args.att_visual:
        #     self.model, self.processor = AutoModel.from_pretrained(args.pretrain_name)  # gloria-chexpert


        
        # build and load pre-trained MAML model
        if args.maml:
            weight_path = args.data_dir + '/' + args.maml_model_path
            print('load initial weights MAML from: %s' % (weight_path))
            self.maml = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)
        # build and load pre-trained Auto-encoder model
        if args.autoencoder:
            self.ae = Auto_Encoder_Model()
            weight_path = args.data_dir + '/' + args.ae_model_path
            print('load initial weights DAE from: %s' % (weight_path))
            self.ae.load_state_dict(torch.load(weight_path))
            # self.convert = nn.Linear(16384, 64)
            self.aeconvert = nn.Linear(16384, args.v_hidden)
            self.convert768 = nn.Linear(16384, 768)
            self.local_convert=nn.Linear(15360, 768)
            self.globalconvert = nn.Linear(768, args.v_hidden)
            self.localconvert = nn.Linear(768, args.v_hidden)
        # Loading tfidf weighted embedding
        if hasattr(args, 'tfidf'):
            self.w_emb = tfidf_loading(args.tfidf, self.w_emb, args)
            
        # Loading the other net
        if args.other_model:
            pass
        # if self.args.both_trans:
        #     self.tanh_gate = linear(22, 16)
        #     self.sigmoid_gate = linear(22, 16)
        # if self.args.use_report:
        #     self.report_model, self.report_processor = AutoModel.from_pretrained("rrg/biomed-roberta-baseline-mimic")
        #     # self.dictionary=Dictionary.load_from_file(self.args.data_dir + '/dictionary.pkl')
        #     self.report_close_att = BiAttention(383, args.hid_dim, args.hid_dim, args.glimpse)
        #     self.report_open_att = BiAttention(383, args.hid_dim, args.hid_dim, args.glimpse)
        #     self.report_open_resnet = BiResNet(args, dataset,use_report=True)
        #     self.report_close_resnet = BiResNet(args, dataset,use_report=True)

        #     self.all_close_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        #     self.all_open_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        #     self.all_open_resnet = BiResNet(args, dataset)
        #     self.all_close_resnet = BiResNet(args, dataset)
        #     self.report_pooling=nn.AvgPool1d(3, stride=2)

        

    def forward(self, v,v_p, q, q_token, a, answer_target):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ## alter by hxj
            ae_v_emb = self.aeconvert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb

        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.args.other_model:
            pass
        if self.args.gloria_global or self.args.att_visual:
            # batch = self.processor.inference(seq=q, image=v[2])
            batch = self.processor.inference(seq=q, image=v_p)
            if self.args.gloria or self.args.grad_gloria:
                out = self.model(**batch)
            else:
                with torch.no_grad():
                    out = self.model(**batch)


            gloria_w_emb = out['word_embeddings']
            gloria_q_emb = out['sent_embeddings'].unsqueeze(1)
            # q_emb = q_emb.to(self.args.device)
            v_emb_global = out['global_features']
            v_emb_local = out['local_features']

            attention_maps = out['attention_maps']
            #get gloria global and weight_local image feature
            #global
            v_emb_global=v_emb_global.unsqueeze(1)
            v_emb_global=v_emb_global.to(self.args.device)# [48, 1,768])

            #weight_local

            ih, iw = v_emb_local.size(2), v_emb_local.size(3)
            sourceL = ih * iw
            batch_size = v_emb_local.size(0)
            v_emb_local = v_emb_local.view(batch_size, -1, sourceL)

            # sents=out['sents']
            # att_maps = []
            # for j,sent in enumerate(sents):
            #
            #     sent=[w for w in sent if not w.startswith("[")]
            #     q_text=' '.join(sent)
            #     gloria_token = torch.Tensor(self.dataset.dictionary.tokenize(q_text, False))
            #     gloria_token=gloria_token.to(self.args.device)
            #     gloria_token = gloria_token.unsqueeze(0)
            #     gloria_token = gloria_token.unsqueeze(0)
            #     attention_maps[j]=attention_maps[j].view(1,-1,sourceL)
            #     att_map=torch.bmm(gloria_token,attention_maps[j])
            #     att_maps.append(att_map)
            # att_mp = torch.cat((att_maps), 0)

                
            
            


            
            att_maps=[]
            for att_map in attention_maps:
                att_map=torch.mean(torch.tensor(att_map),1,keepdim=True)
                # print('att_map shape:{0}'.format(att_map.shape))
                att_maps.append(att_map)
            att_mp=torch.cat((att_maps),0)
            att_mp = att_mp.view(batch_size, -1, sourceL)

            # print('final att_mp"{0}'.format(att_mp.shape))

            # #set words_num=20
            # att_mp = attention_maps[0]#[1,20,19,19]
            # for i in range(1, len(attention_maps)):
            #     att_mp = torch.cat((att_mp, attention_maps[i]), 0)

            
            
            attnT = torch.transpose(att_mp, 1, 2).contiguous()
            v_emb_local = torch.bmm(v_emb_local, attnT)#[48, 768, 361] [48, 361, 20]

            v_emb_localT = torch.transpose(v_emb_local, 1, 2).contiguous()
            v_emb_local = v_emb_localT.to(self.args.device)  # [48, 20,768])
            # print('v_emb_local shape{0}'.format(v_emb_local.shape))






            if self.args.autoencoder:
                if self.args.gloria_global and not self.args.att_visual:
                    v_emb_gloria = v_emb_global.to(self.args.device)  # [48, 1,768])
                    ae_v_emb = self.convert64(ae_v_emb0).unsqueeze(1)  # [48, 1, 64]
                    v_emb=torch.cat((v_emb_gloria, ae_v_emb), 2)# [48, 1, 832]
                elif not self.args.gloria_global and self.args.att_visual:
                    v_emb_gloria = v_emb_local.to(self.args.device)  # [48, 20,768])
                    ae_v_emb = self.convert768(ae_v_emb0).unsqueeze(1)  # [48, 1, 768]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 1)  # [48, 13, 768]
                elif self.args.gloria_global and self.args.att_visual:
                    v_emb_global=self.globalconvert(v_emb_global.squeeze(1)).unsqueeze(1)# [48, 1,64])
                    v_emb_local = self.localconvert(v_emb_local.squeeze(1)).unsqueeze(1)# [48, 1,64])
                    v_emb_gloria = torch.cat((v_emb_global, v_emb_local), 2)  # [48, 1,128])
                    ae_v_emb = self.aeconvert(ae_v_emb0).unsqueeze(1)  # [48, 1, 64]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 2)  # [48, 1, 192]
                    # print('v_emb shape:{0}'.format(v_emb.shape))

                    '''
                    v_emb_gloria = torch.cat((v_emb_global, v_emb_local), 1)  # [48, 21,768])
                    ae_v_emb = self.convert768(ae_v_emb0).unsqueeze(1)  # [48, 1, 768]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 1)  # [48, 22, 768]
                    '''

                    # if self.args.both_trans:
                    #     v_emb=torch.transpose(v_emb, 1, 2).contiguous()# [48 768,22]
                    #     v_emb = torch.mul(torch.tanh(self.tanh_gate(v_emb)),torch.sigmoid(self.sigmoid_gate(v_emb)))# [48 768,16]
                    #     v_emb = torch.transpose(v_emb, 1, 2).contiguous()  # [48 16,768]




            else:
                if self.args.gloria_global and not self.args.att_visual:
                    v_emb_gloria = v_emb_global.to(self.args.device)  # [48, 1,768])
                    v_emb = v_emb_gloria.to(self.args.device)
                elif not self.args.gloria_global and self.args.att_visual:
                    v_emb_gloria = v_emb_local.to(self.args.device)  # [48, 20,768])
                    v_emb = v_emb_gloria.to(self.args.device)
                elif self.args.gloria_global and self.args.att_visual:
                    v_emb_gloria = torch.cat((v_emb_global, v_emb_local), 1)  ## [48, 13,768])
                    v_emb = v_emb_gloria.to(self.args.device)
        # print('v_emb train shape:{0}'.format(v_emb.shape))

        # get type attention

        if self.args.qcr:
            type_att = self.typeatt(q_token)
        else:
            type_att=None

        # get lextual feature    global 
        w_emb = self.w_emb(q_token)
        
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        # get open & close feature
        # print("#######type_att", type_att)
        v_open, v_close, q_open, q_close, a_open, a_close, typeatt_open, typeatt_close = seperate(v_emb, q_emb, a,
                                                                                                  type_att,
                                                                                                  answer_target,self.args)


        class_pred = {}
        if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset)or('OVQA' in self.args.dataset)or('FREE' in self.args.dataset):
            if self.args.image_classify:
                # v_emb_gloria = v_emb_global.to(self.args.device)  # [48, 1,768])
                # ae_v_emb = self.convert64(ae_v_emb0).unsqueeze(1)  # [48, 1, 64]
                # img_class_emb = torch.cat((v_emb_gloria, ae_v_emb), 2)  # [48, 1, 832]
                img_class_emb = v_emb.reshape(v_emb.size(0), -1)
                # print('img_class_emb shape:{0}'.format(img_class_emb.shape))
                img_class_pred = self.image_classify(img_class_emb.squeeze(1))
                class_pred['img_class_pred'] = img_class_pred
            if self.args.question_classify:
                q_emb_t = q_emb.reshape(q_emb.size(0), -1)
                q_class_pred = self.question_classify(q_emb_t)
                class_pred['q_class_pred'] = q_class_pred


        

        if self.args.use_report:
            
            # img_list=[[i] for i in v_p]
            with torch.no_grad():

                report_batch = self.report_processor.inference(image=v_p)
                report_batch_size = len(report_batch["images"])
                beam_size = 8
                encoder_output, encoder_attention_mask = self.report_model.encode(**report_batch)#encoder_output [48, 49, 768]
                expanded_idx = torch.arange(report_batch_size).view(-1, 1).repeat(1, beam_size).view(-1).cuda()

                # Using huggingface generate method
                hyps = self.report_model.dec.generate(
                    input_ids=torch.ones((len(report_batch["images"]), 1),
                                         dtype=torch.long).cuda() * self.report_model.dec.config.bos_token_id,
                    encoder_hidden_states=encoder_output.index_select(0, expanded_idx),
                    encoder_attention_mask=encoder_attention_mask.index_select(0, expanded_idx),
                    num_return_sequences=1,
                    max_length=self.report_processor.tokenizer_max_len,
                    num_beams=8,
                )
                hyps = [self.report_processor.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                        h in hyps]
                rrg_batch = self.processor.inference(seq=hyps)
                rrg_out = self.model.sentence_emb(**rrg_batch)
                # report_emb=rrg_out['sent_embeddings'].unsqueeze(1)
                report_emb = rrg_out.unsqueeze(1)
                report_emb=self.report_pooling(report_emb)
                # print(report_emb.shape)
                
                report_open_emb,report_closed_emb=seperate_report(report_emb,answer_target)

                #fusion qestion and report
                # att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
                att_close1, _ = self.report_close_att(report_closed_emb, q_close)
                att_open1, _ = self.report_open_att(report_open_emb, q_open)

                # bilinear residual network
                # last_output = self.bi_resnet(v_emb,q_emb,att_p)
                output_close1 = self.report_close_resnet(report_closed_emb, q_close, att_close1)#[31, 1024]
                output_open1 = self.report_open_resnet(report_open_emb, q_open, att_open1)
                


                # diverse Attention -> (open + close)
                # att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
                att_close2, _ = self.all_close_att(v_close, output_close1.unsqueeze(1))
                att_open2, _ = self.all_open_att(v_open, output_open1.unsqueeze(1))

                # bilinear residual network
                # last_output = self.bi_resnet(v_emb,q_emb,att_p)
                last_output_close = self.all_close_resnet(v_close, output_close1.unsqueeze(1), att_close2)#[31, 1024]
                last_output_open = self.all_open_resnet(v_open, output_open1.unsqueeze(1), att_open2)

        else:
            # print(type(v_close), type(q_close))
            # print(v_close.shape, q_close.shape)
            # diverse Attention -> (open + close)
            # att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
            # print("######typeatt_close is not None", type(v_close), torch.numel(v_close)==0, v_close is not None )
            if v_close.size(0)!=0:
                att_close, _ = self.close_att(v_close, q_close)

            att_open, _ = self.open_att(v_open, q_open)

            # bilinear residual network
            # last_output = self.bi_resnet(v_emb,q_emb,att_p)
            
            if v_close.size(0)!=0:
                last_output_close = self.close_resnet(v_close, q_close, att_close)#[28, 1024]
            else:
                last_output_close = None
            last_output_open = self.open_resnet(v_open, q_open, att_open)

            # print("#####v_close.size(0)", v_close.size(0), last_output_close)









        # print('#####v_close.size(0)')
        #type attention (5.19 try)
        if typeatt_close!=None and typeatt_open!=None:
            last_output_close = last_output_close * typeatt_close
            last_output_open = last_output_open * typeatt_open

        if self.args.autoencoder and self.args.gloria:
            return last_output_close, last_output_open, a_close, a_open, decoder, out['loss'],class_pred
        elif self.args.autoencoder and not self.args.gloria:
            return last_output_close, last_output_open, a_close, a_open, decoder,class_pred
        elif not self.args.autoencoder and self.args.gloria:
            return last_output_close, last_output_open, a_close, a_open, out['loss'],class_pred
        else:
            return last_output_close, last_output_open, a_close, a_open,class_pred

    def classify(self, close_feat, open_feat):
        if close_feat is not None and open_feat is not None:
            return self.close_classifier(close_feat), self.open_classifier(open_feat)
        elif close_feat is None and open_feat is not None:
            return None, self.open_classifier(open_feat)
        elif close_feat is not None and open_feat is None:
            return self.close_classifier(close_feat), None
        else:
            return None, None

    def image_classify(self, v_emb):
        return self.image_classifier(v_emb)

    def question_classify(self, q_emb):
        return self.question_classifier(q_emb)




    
    def forward_classify(self,v,v_p, q, q_token,a,classify,qid):
        # get visual feature
        
        if self.args.maml:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.aeconvert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder:
            ## alter by hxj
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.args.other_model:
            pass

        if self.args.gloria_global or self.args.att_visual:
            batch = self.processor.inference(seq=q, image=v_p)
            if self.args.gloria:
                out = self.model(**batch)
            else:
                with torch.no_grad():
                    out = self.model(**batch)

            gloria_w_emb = out['word_embeddings']
            gloria_q_emb = out['sent_embeddings'].unsqueeze(1)
            # q_emb = q_emb.to(self.args.device)
            v_emb_global = out['global_features']
            v_emb_local = out['local_features']
            attention_maps = out['attention_maps']
            # get gloria global and weight_local image feature
            # global
            v_emb_global = v_emb_global.unsqueeze(1)
            v_emb_global = v_emb_global.to(self.args.device)  # [48, 1,768])

            # weight_local

            #mean Att
            # att_maps = []
            # for att_map in attention_maps:
            #     att_map = torch.mean(torch.tensor(att_map), 1, keepdim=True)
            #     att_maps.append(att_map)
            # att_mp = torch.cat((att_maps), 0)
            '''
            #words_num=20
            att_mp = attention_maps[0]
            for i in range(1, len(attention_maps)):
                att_mp = torch.cat((att_mp, attention_maps[i]), 0)
            '''

            ih, iw = v_emb_local.size(2), v_emb_local.size(3)
            sourceL = ih * iw
            batch_size = v_emb_local.size(0)
            v_emb_local = v_emb_local.view(batch_size, -1, sourceL)

            att_maps = []
            for att_map in attention_maps:
                att_map = torch.mean(torch.tensor(att_map), 1, keepdim=True)
                # print('att_map shape:{0}'.format(att_map.shape))
                att_maps.append(att_map)
            att_mp = torch.cat((att_maps), 0)
            att_mp = att_mp.view(batch_size, -1, sourceL)

            # sents = out['sents']
            # att_maps = []
            # for j, sent in enumerate(sents):
            #     sent = [w for w in sent if not w.startswith("[")]
            #     q_text = ' '.join(sent)
            #     gloria_token = torch.Tensor(self.dataset.dictionary.tokenize(q_text, False))
            #     gloria_token = gloria_token.to(self.args.device)
            #     gloria_token = gloria_token.unsqueeze(0)
            #     gloria_token = gloria_token.unsqueeze(0)
            #     attention_maps[j] = attention_maps[j].view(1, -1, sourceL)
            #     att_map = torch.bmm(gloria_token, attention_maps[j])
            #     # print('test att maps shape{0}'.format(att_map.shape))
            #     att_maps.append(att_map)
            # att_mp = torch.cat((att_maps), 0)


            attnT = torch.transpose(att_mp, 1, 2).contiguous()
            v_emb_local = torch.bmm(v_emb_local, attnT)
            v_emb_localT = torch.transpose(v_emb_local, 1, 2).contiguous()
            v_emb_local = v_emb_localT.to(self.args.device)  # [48, 20,768])

            if self.args.autoencoder:
                if self.args.gloria_global and not self.args.att_visual:
                    v_emb_gloria = v_emb_global.to(self.args.device)  # [48, 1,768])
                    ae_v_emb = self.convert64(ae_v_emb0).unsqueeze(1)  # [48, 1, 64]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 2)  # [48, 1, 832]
                elif not self.args.gloria_global and self.args.att_visual:
                    v_emb_gloria = v_emb_local.to(self.args.device)  # [48, 20,768])
                    ae_v_emb = self.convert768(ae_v_emb0).unsqueeze(1)  # [48, 1, 768]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 1)  # [48, 13, 768]
                elif self.args.gloria_global and self.args.att_visual:
                    v_emb_global = self.globalconvert(v_emb_global.squeeze(1)).unsqueeze(1)  # [48, 1,64])
                    v_emb_local = self.localconvert(v_emb_local.squeeze(1)).unsqueeze(1)  # [48, 1,64])
                    v_emb_gloria = torch.cat((v_emb_global, v_emb_local), 2)  # [48, 1,128])
                    ae_v_emb = self.aeconvert(ae_v_emb0).unsqueeze(1)  # [48, 1, 64]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 2)  # [48, 1, 192]
                    # print('test v_emb shape{0}'.format(v_emb.shape))
                    '''
                    v_emb_gloria = torch.cat((v_emb_global, v_emb_local), 1)  # [48, 21,768])
                    ae_v_emb = self.convert768(ae_v_emb0).unsqueeze(1)  # [48, 1, 768]
                    v_emb = torch.cat((v_emb_gloria, ae_v_emb), 1)  # [48, 22, 768]
                    '''

                    # if self.args.both_trans:
                    #     v_emb=torch.transpose(v_emb, 1, 2).contiguous()# [48 768,22]
                    #     v_emb = torch.mul(torch.tanh(self.tanh_gate(v_emb)),torch.sigmoid(self.sigmoid_gate(v_emb)))# [48 768,16]
                    #     v_emb = torch.transpose(v_emb, 1, 2).contiguous()  # [48 16,768]
                        
            else:
                if self.args.gloria_global and not self.args.att_visual:
                    v_emb_gloria = v_emb_global.to(self.args.device)  # [48, 1,768])
                    v_emb = v_emb_gloria.to(self.args.device)
                elif not self.args.gloria_global and self.args.att_visual:
                    v_emb_gloria = v_emb_local.to(self.args.device)  # [48, 20,768])
                    v_emb = v_emb_gloria.to(self.args.device)
                elif self.args.gloria_global and self.args.att_visual:
                    v_emb_gloria = torch.cat((v_emb_global, v_emb_local), 1)  ## [48, 13,768])
                    v_emb = v_emb_gloria.to(self.args.device)
        # print('v_emb test shape:{0}'.format(v_emb.shape))

        # get type attention
        if self.args.qcr:
            type_att = self.typeatt(q_token)
        else:
            type_att = None

        # get lextual feature    global 
        w_emb = self.w_emb(q_token)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # get open & close feature
        answer_target = classify(q_token)
       
        _, predicted = torch.max(answer_target, 1)
        v_open, v_close, q_open, q_close, a_open, a_close, typeatt_open, typeatt_close,qid_open,qid_close = seperate_test(v_emb, q_emb, a,
                                                                                                  type_att, predicted,self.args,qid)

        class_pred = {}
        if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset)or ('FREE' in self.args.dataset):
            if self.args.question_classify:
                q_emb_t = q_emb.reshape(q_emb.size(0), -1)
                q_class_pred = self.question_classify(q_emb_t)
                class_pred['q_class_pred'] = q_class_pred
            if self.args.image_classify:
                # v_emb_gloria = v_emb_global.to(self.args.device)  # [48, 1,768])
                # ae_v_emb = self.convert64(ae_v_emb0).unsqueeze(1)  # [48, 1, 64]
                # img_class_emb = torch.cat((v_emb_gloria, ae_v_emb), 2)  # [48, 1, 832]
                img_class_emb = v_emb.reshape(v_emb.size(0), -1)
                img_class_pred = self.image_classify(img_class_emb.squeeze(1))
                class_pred['img_class_pred'] = img_class_pred

        if self.args.use_report:
            # img_list=[[i] for i in v_p]
            with torch.no_grad():

                report_batch = self.report_processor.inference(image=v_p)
                report_batch_size = len(report_batch["images"])
                beam_size = 8
                encoder_output, encoder_attention_mask = self.report_model.encode(
                    **report_batch)  # encoder_output [48, 49, 768]
                expanded_idx = torch.arange(report_batch_size).view(-1, 1).repeat(1, beam_size).view(-1).cuda()

                # Using huggingface generate method
                hyps = self.report_model.dec.generate(
                    input_ids=torch.ones((len(report_batch["images"]), 1),
                                         dtype=torch.long).cuda() * self.report_model.dec.config.bos_token_id,
                    encoder_hidden_states=encoder_output.index_select(0, expanded_idx),
                    encoder_attention_mask=encoder_attention_mask.index_select(0, expanded_idx),
                    num_return_sequences=1,
                    max_length=self.report_processor.tokenizer_max_len,
                    num_beams=8,
                )
                hyps = [self.report_processor.tokenizer.decode(h, skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False) for
                        h in hyps]
                rrg_batch = self.processor.inference(seq=hyps)
                rrg_out = self.model.sentence_emb(**rrg_batch)
                # report_emb=rrg_out['sent_embeddings'].unsqueeze(1)
                report_emb = rrg_out.unsqueeze(1)
                report_emb = self.report_pooling(report_emb)
                # print(report_emb.shape)

                report_open_emb, report_closed_emb = seperate_report(report_emb, predicted)

                # fusion qestion and report
                # att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
                att_close1, _ = self.report_close_att(report_closed_emb, q_close)
                att_open1, _ = self.report_open_att(report_open_emb, q_open)

                # bilinear residual network
                # last_output = self.bi_resnet(v_emb,q_emb,att_p)
                output_close1 = self.report_close_resnet(report_closed_emb, q_close, att_close1)  # [31, 1024]
                output_open1 = self.report_open_resnet(report_open_emb, q_open, att_open1)

                # diverse Attention -> (open + close)
                # att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
                att_close2, _ = self.all_close_att(v_close, output_close1.unsqueeze(1))
                att_open2, _ = self.all_open_att(v_open, output_open1.unsqueeze(1))

                # bilinear residual network
                # last_output = self.bi_resnet(v_emb,q_emb,att_p)
                last_output_close = self.all_close_resnet(v_close, output_close1.unsqueeze(1), att_close2)  # [31, 1024]
                last_output_open = self.all_open_resnet(v_open, output_open1.unsqueeze(1), att_open2)


        else:
            # diverse Attention -> (open + close)
            # att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
            att_close, _ = self.close_att(v_close, q_close)
            att_open, _ = self.open_att(v_open, q_open)

            # bilinear residual network
            # last_output = self.bi_resnet(v_emb,q_emb,att_p)
            last_output_close = self.close_resnet(v_close, q_close, att_close)  # [28, 1024]
            last_output_open = self.open_resnet(v_open, q_open, att_open)



        # type attention (5.19 try)
        if typeatt_close != None and typeatt_open != None:
            last_output_close = last_output_close * typeatt_close
            last_output_open = last_output_open * typeatt_open

        if self.args.autoencoder and self.args.gloria:
            return last_output_close, last_output_open, a_close, a_open, decoder, out['loss'],class_pred,qid_open,qid_close
        elif self.args.autoencoder and not self.args.gloria:
            return last_output_close, last_output_open, a_close, a_open, decoder,class_pred,qid_open,qid_close
        elif not self.args.autoencoder and self.args.gloria:
            return last_output_close, last_output_open, a_close, a_open, out['loss'],class_pred,qid_open,qid_close
        else:
            return last_output_close, last_output_open, a_close, a_open,class_pred,qid_open,qid_close
        


