import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TRANSFORMERS_OFFLINE"] = '1'


import argparse
from dataset_ALL import VQAFeatureDataset
from tools.create_dictionary import Dictionary
from torch.utils.data import DataLoader
import utils
from multi_level_model_update import BAN_Model
import torch
from train_all import train,evaluate
from classify_question import classify_model
import random
# from random import randrange
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    parser.add_argument('--name', type=str, default="48batch_lre3_200epoch",
                        help='description of the train model')
    # GPU config
    parser.add_argument('--seed', type=int, default=1024
                        , help='random seed for gpu.default:1024')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')

    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models',
                        help='save file directory')

    # Training testing or sampling Hyper-parameters
    parser.add_argument('--epochs', type=int, default=1,
                        help='the number of epoches')
    parser.add_argument('--lr', default=0.001, type=float, metavar='lr',
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # # Train with RAD
    parser.add_argument('--use_data', action='store_true', default=True,
                         help='Using TDIUC dataset to train')
    parser.add_argument('--data_dir', type=str,
                         help='RAD dir')
    parser.add_argument('--dataset', type=str, default='RAD',
                        help=['RAD', 'SLAKE'])

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu','sigmoid'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Attention --------------------------------------------------------------------------------------------------------
    # Choices of attention models
    parser.add_argument('--attention', type=str, default='BAN', choices=['BAN'],
                        help='the model we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--glimpse', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Question ---------------------------------------------------------------------------------------------------------
    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='GRU', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--cat', type=bool, default=True,
                        help='concatenated 600-D word embedding')
    parser.add_argument('--hid_dim', type=int, default=1024,
                        help='dim of joint semantic features')

    # Vision -----------------------------------------------------------------------------------------------------------
    # Input visual feature dimension
    parser.add_argument('--v_dim', default=64, type=int,
                        help='visual feature dim')
    parser.add_argument('--v_hidden', default=64, type=int,
                        help='visual hidden feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # Gloria component hyper-parameters
    parser.add_argument('--gloria', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--gloria_alpha', default=0.000001, type=float, metavar='gloria_alpha',
                        help='gloria_alpha')
    parser.add_argument('--att_visual', action='store_true', default=False,
                        help='use Attention weighted local image representation or not')
    parser.add_argument('--gloria_global', action='store_true', default=False,
                        help='use gloria global image feature or not')
    parser.add_argument('--pretrain_name', type=str, default='selfsup/gloria-chexpert',
                        help='[selfsup/gloria-chexpert,selfsup/gloria-mimic-48,selfsup/convirt-mimic]')
    parser.add_argument('--grad_gloria', action='store_true', default=False,
                        help='End to end model?')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')
    
    # other model hyper-parameters
    parser.add_argument('--other_model', action='store_true', default=False,
                        help='End to end model')
    #lr_schedule
    parser.add_argument('--lr_schedule', action='store_true', default=False,
                        help='use lr_schedule or not')
    parser.add_argument('--patience', type=int, default=5,
                        help='lr schedule patience')
    parser.add_argument('--early_stop_limit', type=int, default=20,
                        help='early_stop_limit')
    
    # details
    parser.add_argument('--details',type=str,default='original ')
    # QCR
    parser.add_argument('--qcr', action='store_true', default=False,
                        help='use QCR module or not?')
    # classify image and question
    parser.add_argument('--image_classify', action='store_true', default=False,
                        help='classify image or not')
    parser.add_argument('--img_alpha', default=0.0001, type=float, metavar='img_alpha',
                        help='img_alpha')
    parser.add_argument('--question_classify', action='store_true', default=False,
                        help='classify question type or not')
    parser.add_argument('--q_alpha', default=0.0001, type=float, metavar='q_alpha',
                        help='q_alpha')
    parser.add_argument('--use_report', action='store_true', default=False,
                        help='use_report or not')
    parser.add_argument('--both_trans', action='store_true', default=False,
                        help='both_trans or not')
    # parser.add_argument('--rrg', action='store_true', default=False,
    #                     help='pretrain report generation?')
    parser.add_argument('--record_id',type=int,default=1)


    args = parser.parse_args()
    return args

def seed_torch(seed=855066):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    args = parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    # data = root + '/data_' + args.dataset
    # args.data_dir = data
    # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    # # seed=440300
    # seed = randrange(100000, 999999)
    # args.seed = seed
    seed_torch(args.seed)
    # create word dictionary from train+val dataset
    d = Dictionary.load_from_file(args.data_dir + '/dictionary.pkl')

    # prepare the dataloader
    train_dataset = VQAFeatureDataset('train', args, d, dataroot=args.data_dir)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2, drop_last=False,
                              pin_memory=True)

    # create VQA model and question classify model
    model = BAN_Model(train_dataset, args)
    # print("#####", args.data_dir)
    question_classify = classify_model(d.ntoken,args.data_dir+'/glove6b_init_300d.npy')
    # load the model
    if args.dataset == "SLAKE":
        print('loading slake type classifier model.....')
        ckpt = './saved_models/type_classifier_slake.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')['model_state']
    elif args.dataset == "PATH":
        print('loading pathVQA type classifier model.....')
        ckpt = './saved_models/type_classifier_path.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')['model_state']
    elif args.dataset == "OVQA":
        print('loading OVQA type classifier model.....')
        ckpt = './saved_models/type_classifier_ovqa.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')['model_state']
    elif args.dataset == "Med-2019":
        print('loading pathVQA type classifier model.....')
        ckpt = './saved_models/type_classifier_med2019.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')['model_state']
    else:
        ckpt = './saved_models/type_classifier.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')

    question_classify.load_state_dict(pretrained_model)
    
    # load snapshot
    if args.input is not None:
        # print('loading %s' % args.input)
        pre_ckpt = torch.load(args.input)
        model.load_state_dict(pre_ckpt.get('model_state', pre_ckpt))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(pre_ckpt.get('optimizer_state', pre_ckpt))
        epoch = pre_ckpt['epoch'] + 1

    eval_dset = VQAFeatureDataset('test', args, d, dataroot=args.data_dir)
    # eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    #                          collate_fn=utils.trim_collate)
    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                             collate_fn=utils.trim_collate)

    # training phase
    train(args, model, question_classify, train_loader, eval_loader)

    logger = utils.Logger(os.path.join('./saved_models/', 'test_demo.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())
    evaluate(model, eval_loader, args, logger)

