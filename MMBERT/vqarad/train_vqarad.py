import argparse
from utils_vqarad import seed_everything, Model, VQAMed, train_one_epoch, validate, test, load_data, LabelSmoothing
#import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import sys 
sys.path.append("..") 
sys.path.append("/home/coder/projects/Demo/")
from mysql_connection import connect
# from torchtoolbox.transform import Cutout
# import albumentations as albu
# from albumentations.core.composition import OneOf
# from albumentations.pytorch.transforms import ToTensorV2
import os
import warnings

warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Finetune on VQARAD")

    parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")
    parser.add_argument('--data_dir', type = str, required = False, default = "/home/coder/projects/Med-VQA/data", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "/home/coder/projects/MMBERT/ROCO/roco/roco-dataset-master/data/medvqa/Weights/roco_mlm/val_loss_3.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "/home/coder/projects/MMBERT/vqarad", help = "path to save weights")
    parser.add_argument('--question_type', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 1, help = "num epochs to train")
        # parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default = 16, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-4, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0.3, help = "hidden dropout probability")
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 768, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--record_id', type=int, help='record')


    args = parser.parse_args()

    #wandb.init(project='imageclef20vqa', name = args.run_name, config = args)

    seed_everything(args.seed)


    train_df, test_df = load_data(args)

    if args.question_type:
            
        train_df = train_df[train_df['question_type']==args.question_type].reset_index(drop=True)
        val_df = val_df[val_df['question_type']==args.question_type].reset_index(drop=True)
        test_df = test_df[test_df['question_type']==args.question_type].reset_index(drop=True)


    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df['answer'] = df['answer'].str.lower()
    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes


    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    model = Model(args)
    
    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir))
    
    
    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
        
    model.to(device)

    #wandb.watch(model, log='all')

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)

    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()

    
    train_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomResizedCrop(224,scale=(0.5,1.0),ratio=(0.75,1.333)),
                                    transforms.RandomRotation(10),
                                    # Cutout(),
                                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    



    # traindataset = VQAMed(train_df, imgsize = args.image_size, tfm = train_tfm, args = args)
    # valdataset = VQAMed(val_df, imgsize = args.image_size, tfm = val_tfm, args = args)
    # testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args)


    temp = round(len(train_df)*0.9)
    val_df = train_df[temp:]
    train_df = train_df[:temp]
    # 测试代码记得修改，删掉
    # train_df = train_df[:round(temp*0.2)]
    # print("######", len(val_df))
    
    traindataset = VQAMed(train_df, tfm = train_tfm, args = args)
    # valdataset = VQAMed(val_df, tfm = train_tfm, args = args)
    valdataset = VQAMed(val_df, tfm = val_tfm, args = args)
    testdataset = VQAMed(test_df, tfm = test_tfm, args = args)

    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    # print("######"len(trainloader))
    valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    val_best_acc = 0
    test_best_acc = 0
    best_loss = np.inf
    counter = 0

    conn=connect()
    cursor=conn.cursor()

    for epoch in range(args.epochs):

        print(f'Epoch {epoch+1}/{args.epochs}')


        train_loss, train_acc = train_one_epoch(trainloader, model, optimizer, criterion, device, scaler, args, train_df,idx2ans)
        val_loss, val_predictions, val_acc, val_bleu = validate(valloader, model, criterion, device, scaler, args, val_df,idx2ans)
        test_loss, test_predictions, test_acc,test_bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)

        scheduler.step(train_loss)

        log_dict = test_acc

        for k,v in test_acc.items():
            log_dict[k] = v

        log_dict['train_loss'] = train_loss
        log_dict['test_loss'] = test_loss
        log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]

        #wandb.log(log_dict)

        content = f'Learning rate: {(optimizer.param_groups[0]["lr"]):.7f}, Train loss: {(train_loss):.4f}, Train acc: {train_acc},Test loss: {(test_loss):.4f},  Test acc: {test_acc}'
        print(content)

        sql="INSERT INTO detail (batch_size, epoch,loss,record_id) values(%s, %s, %s, %s)" %  (args.batch_size,epoch+1,train_loss,args.record_id)     
        cursor.execute(sql)
        conn.commit()
            
        if test_acc['total_acc'] > test_best_acc:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
                torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}_test_acc.pt'))
            test_best_acc=test_acc['total_acc']
    test_loss, test_predictions, test_acc,bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
    print("final:")
    closed,open,all=np.round(test_acc['closed_acc'],1),np.round(test_acc['open_acc'],1),np.round(test_acc['total_acc'],1)
    print(closed,open,all)
    sql="UPDATE `record` SET closed=%f,open=%f,`all`=%f,`status`='%s' where id=%s" % (closed,open,all,'complete',args.record_id)
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()
    print(bleu)