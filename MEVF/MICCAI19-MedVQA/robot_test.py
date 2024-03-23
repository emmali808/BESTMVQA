"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
import base_model
import utils
import pandas as pd
import os
import json
import sys
sys.path.append("/home/coder/projects/Demo/")
from mysql_connection import connect


answer_types = ['CLOSED', 'OPEN', 'ALL']
quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/SAN_MEVF',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=int, default=19,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--RAD_dir', type=str,
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')

    ## alter by hxj
    parser.add_argument('--img_size', default=84, type=int, help='image size')

    parser.add_argument('--record_id', type=int, help='record', default=1)

    parser.add_argument('--question', type = str, required = False, default = "what modality is shown?", help = "predict question")
    parser.add_argument('--image_path', type = str, required = False, default = "/home/coder/projects/SystemDataset/robot/synpic54082.jpg", help = "predict image path")
    parser.add_argument('--robot', default=False, help='robot')
    # Return args
    args = parser.parse_args()
    return args
# Load questions
def get_question(q, dataloader):
    q = q.squeeze(0)
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)

# Load answers
def get_answer(p, dataloader):
    _m, idx = p.max(1)
    return dataloader.dataset.label2ans[idx.item()]

# Logit computation (for train, test or evaluate)
def get_result(model, dataloader, device, args,trainloader):
    if "SLAKE" in args.RAD_dir:
        quesntion_types = ['Modality', 'Organ', 'Size', 'Abnormality', 'Position', 'Quantity', 'Plane', 'Shape', 'Color', 'KG']
        answer_types = ['CLOSED', 'OPEN', 'ALL']
    elif "PATH" in args.RAD_dir:
        # quesntion_types = ['where', 'what', 'are', 'does', 'how', 'do', 'is', 'when', 'have', 'did', 'infiltration', 'by', 
                            # 'metastatic', 'microscopy', 'why', 'there', 'were', 'one', 'the', 'dose', 'has', 'b', ]
        answer_types = ['other', 'yes/no', 'all', 'number']
    elif "RAD" in args.RAD_dir:
        answer_types = ['CLOSED', 'OPEN', 'ALL']
        quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']         
                    
    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent']
    if "PATH" not in args.RAD_dir:
        question_types_result = dict((i, dict((j, dict((k, 0.0) for k in keys)) for j in quesntion_types)) for i in answer_types)
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    with torch.no_grad():
        for v, q, a, ans_type, q_types, p_type in iter(dataloader):
            
            # if "RAD" in args.RAD_dir:
            #     if p_type[0] != "freeform":
            #         continue
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            q = q.to(device)
            a = a.to(device)
            # inference and get logit
            if args.autoencoder:
                features, _ = model(v, q)
            else:
                features = model(v, q)
            preds = model.classifier(features)
            # print(preds)
            final_preds = preds
            # print("#######", ans_type) 

            # print("---------*****----------")
            print(get_answer(final_preds,dataloader))
            # print("###########", a.data.max(1))

            # print("###########", trainloader.dataset.ans2label['yes'])
            # print("###########", trainloader.dataset.label2ans[3])

            # print("-----------------------")
            # print(get_answer(a.data,dataloader))
            batch_score = compute_score_with_logits(final_preds, a.data).sum()

            # Compute accuracy for each type answer
            # print("####", ans_type[0])
            if "PATH" in args.RAD_dir:
                result[ans_type[0]]['count'] += 1.0
                result[ans_type[0]]['true'] += float(batch_score)
                result[ans_type[0]]['real'] += float(a.sum())                
                result['all']['count'] += 1.0
                result['all']['true'] += float(batch_score)
                result['all']['real'] += float(a.sum())   

                q_types = q_types[0].split(", ")
            else:
                result[ans_type[0]]['count'] += 1.0
                result[ans_type[0]]['true'] += float(batch_score)
                result[ans_type[0]]['real'] += float(a.sum())
                result['ALL']['count'] += 1.0
                result['ALL']['true'] += float(batch_score)

                q_types = q_types[0].split(", ")

                for i in q_types:
                    question_types_result[ans_type[0]][i]['count'] += 1.0
                    question_types_result[ans_type[0]][i]['true'] += float(batch_score)
                    question_types_result[ans_type[0]][i]['real'] += float(a.sum())

                    question_types_result['ALL'][i]['count'] += 1.0
                    question_types_result['ALL'][i]['true'] += float(batch_score)
                    question_types_result['ALL'][i]['real'] += float(a.sum())

        if "PATH" in args.RAD_dir:
            for i in answer_types:
                result[i]['score'] = result[i]['true']/result[i]['count']
                result[i]['score_percent'] = round(result[i]['score']*100,1)
            return result, None            
        else:
            for i in answer_types:
                result[i]['score'] = result[i]['true']/result[i]['count']
                result[i]['score_percent'] = round(result[i]['score']*100,1)

                for j in quesntion_types:
                    if question_types_result[i][j]['count'] != 0.0:
                        question_types_result[i][j]['score'] = question_types_result[i][j]['true'] / question_types_result[i][j]['count']
                        question_types_result[i][j]['score_percent'] = round(question_types_result[i][j]['score']*100, 1)
                    if question_types_result[i][j]['real'] != 0.0:
                        question_types_result[i][j]['real_percent'] = round(question_types_result[i][j]['real']/question_types_result[i][j]['count']*100.0, 1)
            return result, question_types_result

# Test phase
if __name__ == '__main__':
    args = parse_args()
    # print(args)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Check if evaluating on TDIUC dataset or VQA dataset
    if args.use_RAD:
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir , 'dictionary.pkl'))
        eval_dset = dataset_RAD.VQAFeatureDataset(args.split, args, dictionary)
        train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    # print(model)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=utils.trim_collate)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)

    def save_questiontype_results(outfile_path, quesntion_types_result):
        for i in quesntion_types_result:
            pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')
    # Testing process
    def process(args, model, eval_loader,train_loader):
        model_path = args.input + '/model_epoch%s.pth' % args.epoch
        print('loading %s' % model_path)
        model_data = torch.load(model_path)
        CLOSED_acc,OPEN_acc,ALL_acc=0.0,0.0,0.0
        # Comment because do not use multi gpu
        # model = nn.DataParallel(model)
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        if args.use_RAD:
            result, quesntion_types_result = get_result(model, eval_loader, args.device, args,train_loader)
            outfile_path = args.output + '/' + args.input.split('/')[1]
            outfile = outfile_path + '/results.json'
            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            print(result)
            print(quesntion_types_result)
            json.dump(result, open(outfile, 'w'))
            if "PATH" not in args.RAD_dir: 
                save_questiontype_results(outfile_path, quesntion_types_result)
                #CLOSED_acc=(result['yes/no']['true']+result['number']['true'])/(result['yes/no']['count']+result['number']['count'])
                # return 0.0,result['other']['score_percent'],result['all']['score_percent']
            # CLOSED_acc,OPEN_acc,ALL_acc=result['CLOSED']['score_percent'],result['OPEN']['score_percent'],result['ALL']['score_percent']
            if 'PATH' in args.RAD_dir:
                CLOSED_acc=round((result['yes/no']['true']+result['number']['true'])/(result['yes/no']['count']+result['number']['count'])*100,1)
                OPEN_acc=result['other']['score_percent']
                ALL_acc=result['all']['score_percent']
            else:
                CLOSED_acc,OPEN_acc,ALL_acc=result['CLOSED']['score_percent'],result['OPEN']['score_percent'],result['ALL']['score_percent']
        return CLOSED_acc,OPEN_acc,ALL_acc
    CLOSED_acc,OPEN_acc,ALL_acc=process(args, model, eval_loader,train_loader)
    print(CLOSED_acc,OPEN_acc,ALL_acc)
    conn = connect()
    # 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
    cursor = conn.cursor()
    # 使用 execute()  方法执行 SQL 查询
    sql="UPDATE `record` SET closed=%f,open=%f,`all`=%f,`status`='%s' where id=%s" % (CLOSED_acc,OPEN_acc,ALL_acc,'complete',args.record_id)
    cursor.execute(sql)
    conn.commit()
    # 关闭数据库连接
    cursor.close()
    conn.close()
