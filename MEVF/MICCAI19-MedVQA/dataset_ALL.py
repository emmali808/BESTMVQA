# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         vqa_dataset
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/5/1
#-------------------------------------------------------------------------------


"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import torch
from language_model import WordEmbedding
from torch.utils.data import Dataset,DataLoader
import itertools
import warnings
import h5py
from PIL import Image

import argparse
import torchvision.transforms as transforms
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def _create_entry(dataroot,img, data, answer,organ_list,question_type_list):
    if None != answer:
        if ('RAD' in dataroot) or ('FREE' in dataroot):
            answer.pop('image_name')
            answer.pop('qid')
        elif ('SLAKE' in dataroot) or ('PATH' in dataroot)or ('OVQA' in dataroot):
            answer.pop('img_name')
            answer.pop('qid')

    if ('RAD' in dataroot) or ('SLAKE' in dataroot) or ('FREE' in dataroot):
        organ2idx = {organ: idx for idx, organ in enumerate(organ_list)}
        question_type2idx = {q_t: idx for idx, q_t in enumerate(question_type_list)}
        entry = {
            'qid': data['qid'],
            'image_name': data['image_name'],
            'image': img,
            'image_organ': organ2idx[data['image_organ'].upper()],
            'question': data['question'],
            'answer': answer,
            'answer_type': data['answer_type'],
            'question_type': question_type2idx[data['question_type'].replace(' ', '').split(',')[0].upper()],
            # 'phrase_type' : data['phrase_type']
        }
        # print(data['qid'])
        return entry
    elif 'PATH'in dataroot:
       
        question_type2idx = {q_t: idx for idx, q_t in enumerate(question_type_list)}
        entry = {
            'qid': data['qid'],
            'image_name': data['image_name'],
            'image': img,
            'question': data['question'],
            'answer': answer,
            'answer_type': data['answer_type'],
            'question_type': question_type2idx[data['question_type'].upper()],
            # 'phrase_type' : data['phrase_type']
        }
        # print(data['qid'])
        return entry
    elif 'OVQA' in dataroot:
        organ2idx = {organ: idx for idx, organ in enumerate(organ_list)}
        question_type2idx = {q_t: idx for idx, q_t in enumerate(question_type_list)}
        entry = {
            'qid': data['qid'],
            'image_name': data['image_name'],
            'image': img,
            'image_organ': organ2idx[data['image_organ'].upper()],
            'question': data['question'],
            'answer': answer,
            'answer_type': data['answer_type'],
            'question_type': question_type2idx[data['question_type'].upper()],
            # 'phrase_type' : data['phrase_type']
        }
        # print(data['qid'])
        return entry
        



    

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError:
    return False
  return True

def _load_dataset(dataroot, name, img_id2val, label2ans,organ_list,question_type_list):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path, encoding='utf-8'))
    if 'SLAKE' in dataroot:
        samples = [sample for sample in samples if sample['q_lang'] == "en"]
    samples = sorted(samples, key=lambda x: x['qid'])
    

    answer_path = os.path.join(dataroot, 'cache', '%s_openclose_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    # if name=='test' and 'FREE' in dataroot:
    #     sampleIds = [sample['qid'] for sample in samples]
    #     answers=[answer for answer in answers if answer['qid'] in sampleIds]
    answers = sorted(answers, key=lambda x: x['qid'])
    
    utils.assert_eq(len(samples), len(answers))
    
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        if ('SLAKE' in dataroot) or ('PATH' in dataroot)or('OVQA' in dataroot):
            utils.assert_eq(sample['image_name'], answer['img_name'])

        else:
            utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']

        
        
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(dataroot,img_id2val[img_id], sample, answer,organ_list,question_type_list))

    return entries

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args,dictionary, dataroot='/home/coder/projects/Med-VQA/data_SLAKE', question_len=12):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        self.dataroot = dataroot
        assert name in ['train', 'test','val']
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        if ('RAD' in dataroot) or ('FREE' in dataroot):
            # self.num_ans_candidates =  487 # 56 431
            self.organ_list = ['CHEST', 'ABD', 'HEAD']
            self.question_type_list = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER',
                                  'ATTRIB']
        elif 'SLAKE' in dataroot:
            # self.num_ans_candidates = 254#219 35
            self.organ_list = ['NECK', 'BRAIN', 'PELVIC CAVITY', 'BRAIN_TISSUE', 'CHEST_MEDIASTINAL', 'BRAIN_FACE',
                          'CHEST_LUNG', 'CHEST_HEART', 'ABDOMEN', 'LUNG']
            self.question_type_list = ['QUANTITY', 'MODALITY', 'SHAPE', 'COLOR', 'SIZE', 'PLANE', 'ABNORMALITY', 'POSITION',
                                  'ORGAN', 'KG']
        elif 'PATH' in dataroot:
            # self.num_ans_candidates = 4122  #  2 4120
            self.organ_list =None
            self.question_type_list = ['ACUTE', 'HAD', 'RETICULIN', 'B', 'A', 'WHOSE', 'IS', 'WHERE', 'WERE', 'INFILTRATION', 'THERE', 'THESE', 'WAS', 'ARE', 'ONE', 'MICROSCOPY', 'THE', 'WHEN', 'HAVE', 'D', 'HAS', 'WHO', 'DOSE', 'CAN', 'WHY', 'WHAT', 'SECTIONED', 'HOW', 'DOES', 'UNDER', 'METASTATIC', 'BY', 'DID', 'IMPAIRED', 'TWO', 'DO']
        elif 'OVQA' in dataroot:#745 322
            self.organ_list = ['CHEST', 'LEG', 'HAND', 'HEAD']
            self.question_type_list =['ABNORMALITY', 'ATTRIBUTE OTHER', 'ORGAN SYSTEM', 'MODALITY', 'PLANE', 'CONDITION']


        # close & open
        self.label2close = cPickle.load(open(os.path.join(dataroot,'cache','close_label2ans.pkl'),'rb'))
        self.label2open = cPickle.load(open(os.path.join(dataroot, 'cache', 'open_label2ans.pkl'), 'rb'))

        self.num_open_candidates = len(self.label2open)
        self.num_close_candidates = len(self.label2close)
        self.num_ans_candidates =self.num_open_candidates +self.num_close_candidates
        # print("close num:{0}".format(self.num_close_candidates))
        # print("all num:{0}".format(self.num_ans_candidates))



        self.args = args



        # End get the number of answer type class
        self.dictionary = dictionary

        # TODO: load img_id2idx
        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans,self.organ_list,self.question_type_list)
        
         # load image data for MAML module
        if args.maml:
            # TODO: load images
            images_path = os.path.join(dataroot, 'images84x84.pkl')
            print('loading MAML image data from file: '+ images_path)
            self.maml_images_data = cPickle.load(open(images_path, 'rb'))
        # load image data for Auto-encoder module
        if args.autoencoder:
            # TODO: load images
            images_path = os.path.join(dataroot, 'images128x128.pkl')
            print('loading DAE image data from file: '+ images_path)
            self.ae_images_data = cPickle.load(open(images_path, 'rb'))
        self.gloria_images_data=self.ae_images_data
            
        
        # tokenization
        
        self.tokenize(question_len)
        self.tensorize()
        if args.autoencoder and args.maml:
            self.v_dim = args.feat_dim * 2
        else:
            self.v_dim = args.feat_dim

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    

    def tensorize(self):
        if self.args.maml:
            if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset)or ('FREE' in self.args.dataset):
                self.maml_images_data = torch.from_numpy(self.maml_images_data)
            elif 'PATH' in self.args.dataset:
                self.maml_images_data = torch.stack(self.maml_images_data)
            self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')
        if self.args.autoencoder:
            if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset)or ('FREE' in self.args.dataset):
                self.ae_images_data = torch.from_numpy(self.ae_images_data)

            elif 'PATH' in self.args.dataset:
                self.ae_images_data = torch.stack(self.ae_images_data)
            self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        for entry in self.entries:
            question = np.array(entry['q_token'])
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        qid=entry['qid']
        question = entry['question']
        question=question.lower()
        question=question.replace('x-ray','xray').replace('x ray','xray')
        question_token = entry['q_token']
        answer = entry['answer']
        type = answer['type']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
       
        image_data = [0, 0]
        image_name = entry['image_name']
        

        

        image_path = os.path.join(self.dataroot, 'images/', image_name)
        if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset) or ('FREE' in self.args.dataset):
            image_organ=entry['image_organ']
        elif 'PATH' in self.args.dataset:
            image_organ=None




        if self.args.maml:
            if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset)or ('FREE' in self.args.dataset):
                maml_images_data = self.maml_images_data[entry['image']].reshape(84*84)
            elif 'PATH' in self.args.dataset:
                maml_images_data = self.maml_images_data[entry['image']].reshape(
                    3 * 84 * 84)
            image_data[0] = maml_images_data
        if self.args.autoencoder:
            ae_images_data = self.ae_images_data[entry['image']].reshape(128*128)
            image_data[1] = ae_images_data

        # gloria_compose = transforms.Compose([
        #     transforms.CenterCrop(size=(224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # 
        # ])
        # gloria_images_data = self.gloria_images_data[entry['image']].reshape(128 * 128)
        # image_data[2]=gloria_images_data

        if ('RAD' in self.args.dataset) or ('SLAKE' in self.args.dataset) or ('OVQA' in self.args.dataset)or ('FREE' in self.args.dataset):
            if answer_type == 'CLOSED':
                answer_target = 0
            else:
                answer_target = 1
        elif 'PATH' in self.args.dataset:
            if answer_type == 'yes/no':
                answer_target = 0
            else:
                answer_target = 1
        
        if image_organ!=None:
            organ_label = np.array(image_organ)
            organ_scores = np.array([1.], dtype=np.float32)
            organ_label = torch.from_numpy(organ_label)
            organ_scores = torch.from_numpy(organ_scores)
            organ_target = torch.zeros(len(self.organ_list))
            organ_target.scatter_(0, organ_label, organ_scores)
        else:
            organ_target=[]


        qt_label = np.array(question_type)
        qt_scores = np.array([1.], dtype=np.float32)
        qt_label = torch.from_numpy(qt_label)
        qt_scores = torch.from_numpy(qt_scores)
        qt_target = torch.zeros(len(self.question_type_list))
        qt_target.scatter_(0, qt_label, qt_scores)

        if None!=answer:
            if ('RAD' in self.args.dataset)or ('FREE' in self.args.dataset):
                labels = answer['labels']
                scores = answer['scores']
                composed_target = torch.zeros(self.num_ans_candidates)  # close + open
                if answer_target == 0:
                    target = torch.zeros(self.num_close_candidates)
                    if labels is not None:
                        target.scatter_(0, labels, scores)
                    composed_target[:self.num_close_candidates] = target
                else:
                    target = torch.zeros(self.num_open_candidates)
                    if labels is not None:
                        target.scatter_(0, labels, scores)
                    composed_target[self.num_close_candidates: self.num_ans_candidates] = target
            elif ('SLAKE' in self.args.dataset) or ('PATH' in self.args.dataset)or ('OVQA' in self.args.dataset) :
                labels = answer['labels']
                scores = answer['scores']
                composed_target = torch.zeros(self.num_ans_candidates)  # close + open
                if answer_target == 0:

                    target = torch.zeros(self.num_close_candidates)
                    if labels is not None:
                        try:
                            target.scatter_(0, labels, scores)
                        except:
                            print('a_t=0 {0},{1}'.format(qid,labels))
                    composed_target[:self.num_close_candidates] = target
                else:

                    target = torch.zeros(self.num_open_candidates)
                    if labels is not None:
                        try:
                            target.scatter_(0, labels - self.num_close_candidates, scores)
                        except:
                            print('a_t=1 {0},{1}'.format(qid,labels))


                    composed_target[self.num_close_candidates: self.num_ans_candidates] = target
            return image_data, image_path, organ_target, question, question_token, composed_target, answer_type, qt_target, answer_target,qid



        else:
            return image_data, image_path, organ_target, question, question_token, answer_type, qt_target, answer_target,qid

    def __len__(self):
        return len(self.entries)

def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    # print(dictionary.word2idx, dictionary.idx2word)
    if args.use_RAD:
        dataroot = args.RAD_dir
    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path,encoding="utf-8"))
            for question in questions:
                ## alter by hxj
                if 'SLAKE' in args.RAD_dir:
                    if question['q_lang'] == "en":
                        populate(inds, df, question['question'])
                else:
                    populate(inds, df, question['question'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # print("#####", tfidf.shape)

    # Latent word embeddings
    emb_dim = 300
    glove_file = os.path.join(dataroot, 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights






if __name__=='__main__':
    # dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')
    # tfidf, weights = tfidf_from_questions(['train'], None, dictionary)
    # w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    # w_emb.init_embedding(os.path.join('data_RAD', 'glove6b_init_300d.npy'), tfidf, weights)
    # with open('data_RAD/embed_tfidf_weights.pkl', 'wb') as f:
    #     torch.save(w_emb, f)
    # print("Saving embedding with tfidf and weights successfully")

    # dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')
    # w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    # with open('data_RAD/embed_tfidf_weights.pkl', 'rb') as f:
    #     w_emb = torch.load(f)
    # print("Load embedding with tfidf and weights successfully")
    #
    # # TODO: load img_id2idx
    # img_id2idx = json.load(open(os.path.join('./data_RAD', 'imgid2idx.json')))
    # label2ans_path = os.path.join('./data_RAD', 'cache', 'trainval_label2ans.pkl')
    # label2ans = cPickle.load(open(label2ans_path, 'rb'))
    # entries = _load_dataset('./data_RAD', 'train', img_id2idx, label2ans)
    # print(entries)

    import main

    args = main.parse_args()

    dataroot = './data'

    d = Dictionary.load_from_file(os.path.join(dataroot,'dictionary.pkl'))
    dataset = VQAFeatureDataset('test',args,d,dataroot)
    train_data = DataLoader(dataset,batch_size=20,shuffle=False,num_workers=2,pin_memory=True,drop_last=False)
    for i,row in enumerate(train_data):
        image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
        print(target.shape)
        break

