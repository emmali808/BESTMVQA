import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# import _init_paths
import torch
# from config import cfg, update_config
# from dataset import *
from dataset_ALL import VQAFeatureDataset
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tools.create_dictionary import Dictionary
# from utils.create_dictionary import Dictionary
from language_model import WordEmbedding,QuestionEmbedding
import argparse
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
import utils
from datetime import datetime
from classify_question import classify_model


def parse_args():
    parser = argparse.ArgumentParser(description="Type classifier")
    # GPU config
    parser.add_argument('--output', type=str, default='type_saved_models',
                        help='save file directory')
    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')
    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')
    parser.add_argument('--dataset', type=str, default='RAD',
                        help=['RAD', 'SLAKE'])
    parser.add_argument('--v_dim', default=64, type=int,
                        help='visual feature dim')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
    args = parser.parse_args()
    return args


# Evaluation
def evaluate(model, dataloader,logger,device):
    score = 0
    number =0
    model.eval()
    with torch.no_grad():
        for i,row in enumerate(dataloader):
            # image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            image_data, image_path, _, question, question_token, composed_target, answer_type, qt_target, answer_target,qid=row
            question_token, answer_target = question_token.to(device), answer_target.to(device)
            output = model(question_token)
            pred = output.data.max(1)[1]
            correct = pred.eq(answer_target.data).cpu().sum()
            score+=correct.item()
            number+=len(answer_target)

        score = score / number * 100.

    logger.info('[Validate] Val_Acc:{:.6f}%'.format(score))
    return score


if __name__=='__main__':
    args = parse_args()
    # update_config(cfg, args)
    # root = os.path.dirname(os.path.abspath(__file__))
    dataroot = ""
    if 'SLAKE' in args.dataset or 'PATH' in args.dataset:
        dataroot = '/home/coder/projects/Med-VQA' + '/data_' + args.dataset
    elif 'Med-2019' in args.dataset:
        dataroot = '/home/coder/projects/MEVF/MICCAI19-MedVQA/data_Med/VQA-Med-2019'
    elif 'OVQA' in args.dataset:
        dataroot = '/home/coder/projects/SystemDataset/data_OVQA_as_RAD'

    # # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # print(device)

    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    d = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))
    train_dataset = VQAFeatureDataset('train',args,d,dataroot=dataroot)
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    val_dataset = VQAFeatureDataset('val',args,d,dataroot=dataroot)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    net = classify_model(d.ntoken, os.path.join(dataroot, 'glove6b_init_300d.npy'))
    net =net.to(device)

    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    # ckpt_path = os.path.join('./log', run_timestamp)
    # utils.create_dir(ckpt_path)
    ckpt_path = './saved_models'
    model_path = ""
    if 'SLAKE' in dataroot:
        model_path = os.path.join(ckpt_path, "type_classifier_slake.pth")
    elif 'PATH' in dataroot:
        model_path = os.path.join(ckpt_path, "type_classifier_path.pth")
    elif 'Med-2019' in dataroot:
        model_path = os.path.join(ckpt_path, "type_classifier_med2019.pth")
    elif 'OVQA' in dataroot:
        model_path = os.path.join(ckpt_path, "type_classifier_ovqa.pth")
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(net)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    ## alter by hxj 测试代码,默认值为100
    epochs = 2
    best_eval_score = 0
    best_epoch = 0
    for epoch in range(epochs):
        net.train()
        acc = 0.
        number_dataset = 0
        total_loss = 0
        for i, row in enumerate(train_data):
            # image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            image_data, image_path, _, question, question_token, composed_target, answer_type, qt_target, answer_target,qid = row
            question_token = question_token.to(device)
            answer_target=answer_target.to(device)
            optimizer.zero_grad()
            output = net(question_token)
            loss = criterion(output,answer_target)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct = (pred==answer_target).data.cpu().sum()
            
            acc += correct.item()
            number_dataset += len(answer_target)
            total_loss+= loss
        
        total_loss /= len(train_data)
        acc = acc/ number_dataset * 100.

        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, acc
                                                                     ))
        # Evaluation
        if val_data is not None:
            eval_score = evaluate(net, val_data, logger, device)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                # utils.save_model(model_path, net, best_epoch, eval_score)
                utils.save_model(model_path, net, best_epoch, optimizer)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))
