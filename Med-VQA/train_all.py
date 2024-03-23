# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/4/8
#-------------------------------------------------------------------------------
import os
import time
import torch
import utils
from datetime import datetime
import torch.nn as nn
from torch.optim import lr_scheduler
from utils import *
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/coder/projects/Demo/")
from mysql_connection import connect
import numpy as np
# from tensorboardX import SummaryWriter
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp
def draw(x_axis_data,train_loss_data,eval_loss_data,f_name,type,args):


    # epoch,train_loss,test_loss
    train_loss_data = torch.tensor(train_loss_data, device='cpu')
    eval_loss_data = torch.tensor(eval_loss_data, device='cpu')
    # 画图
    if args.dataset=='SLAKE':
        plt.plot(x_axis_data[:70], train_loss_data[:70], 'b*--', alpha=0.5, linewidth=1, label='Train')
        plt.plot(x_axis_data[:70], eval_loss_data[:70], 'rs--', alpha=0.5, linewidth=1, label='Eval')
    elif args.dataset == 'RAD':
        plt.plot(x_axis_data[:100], train_loss_data[:100], 'b*--', alpha=0.5, linewidth=1, label='Train')
        plt.plot(x_axis_data[:100], eval_loss_data[:100], 'rs--', alpha=0.5, linewidth=1, label='Eval')
    elif args.dataset == 'FREE':
        plt.plot(x_axis_data[:100], train_loss_data[:100], 'b*--', alpha=0.5, linewidth=1, label='Train')
        plt.plot(x_axis_data[:100], eval_loss_data[:100], 'rs--', alpha=0.5, linewidth=1, label='Eval')
    elif args.dataset == 'PATH':
        plt.plot(x_axis_data[:40], train_loss_data[:40], 'b*--', alpha=0.5, linewidth=1, label='Train')
        plt.plot(x_axis_data[:40], eval_loss_data[:40], 'rs--', alpha=0.5, linewidth=1, label='Eval')




    plt.legend()  # 显示上面的label

    plt.xlabel('Epoch')
    plt.ylabel(type)
    f = plt.gcf()  # 获取当前图像
    f.savefig(f_name)
    f.clear()  # 释放内存

# Train phase
def train(args, model,question_model, train_loader, eval_loader,s_opt=None, s_epoch=0):
    train_loss_data = []
    test_loss_data=[]
    train_acc_data = []
    test_acc_data = []

    device = args.device
    model = model.to(device)
    question_model = question_model.to(device)
    # create packet for output
    utils.create_dir(args.output)
    # for every train, create a packet for saving .pth and .log
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join(args.output,run_timestamp)
    utils.create_dir(ckpt_path)
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())
    logger.info(">>>The GPU is:{0}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(">>>The seed is:{0}".format(args.seed))
    # Adamax optimizer
    optim = torch.optim.Adamax(params=model.parameters(),lr=args.lr)
    # optim = torch.optim.RAdam(params=model.parameters(), lr=args.lr)
    # Scheduler learning rate
    #lr_decay = lr_scheduler.CosineAnnealingLR(optim,T_max=len(train_loader))  # only fit for sgdr
    if args.lr_schedule:
        # Lr Scheduler and early stop
        lr_decay_params = {'factor': 0.75, 'patience': args.patience, 'min_lr': 1e-06, 'threshold': 0.001, 'threshold_mode': 'abs'}
        early_stop_metric = "eval_acc"  # validation_loss,training_loss,eval_acc
        training_scheduler = TrainingScheduler(lr_decay_func="ReduceLROnPlateau",
                                               optimizer=optim,
                                               early_stop_metric=early_stop_metric,
                                               early_stop_limit=args.early_stop_limit,
                                               lr_decay_params=lr_decay_params)
        # logger.settings('Training scheduler created')
        logger.info(training_scheduler)
        eval_start = 0
        decay_metric_start = 0
        early_stop_start = 0
        grad_accu = 1
    num_epochs=args.epochs
    lr_default = args.lr
    lr_decay_step = 2 if num_epochs == 20 else (num_epochs - 10) // 5
    lr_decay_rate = .75
    lr_decay_epochs = range(10, num_epochs, lr_decay_step) if eval_loader is not None else range(10, num_epochs,
                                                                                                 lr_decay_step)


    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    ae_criterion = torch.nn.MSELoss()
    # writer = SummaryWriter(log_dir=ckpt_path)
    update_freq = int(args.update_freq)

    best_eval_score = 0
    best_epoch = 0
    # Epoch passing in training phase
    conn=connect()
    cursor=conn.cursor()
    for epoch in range(s_epoch, args.epochs):
        total_loss = 0
        train_score = 0
        number=0
        num_updates = 0
        
        if epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.info('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        model.train()
        
        # Predicting and computing score
        for i, (v,v_p,image_organ, q,q_token, a, answer_type, question_type, answer_target,qid) in enumerate(train_loader):
            # print("######", answer_target)
            #lr_decay.step()
            optim.zero_grad()

            if args.maml:
                # print("#####", v[0].shape)
                # exit()
                ## alter by hxj
                # if "PATH" in args.dataset:
                #     v[0] = v[0].reshape(v[0].shape[0], 3, 84, 84).unsqueeze(1)
                # else:
                #     v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)

                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)
        
            q_token = q_token.to(device)
            a = a.to(device)
            if ('RAD' in args.dataset) or ('SLAKE' in args.dataset)or ('FREE' in args.dataset):
                image_organ=image_organ.to(device)
            question_type=question_type.to(device)

            # MEVF loss computation
            if args.autoencoder and args.gloria:
                last_output_close, last_output_open, a_close, a_open, decoder, gloria_loss,class_pred = model(v, v_p, q, q_token,
                                                                                                   a, answer_target)
            elif args.autoencoder and not args.gloria:
                # print("#######", np.array(v).shape,  np.array(v_p).shape, np.array(q).shape, (q_token).shape, (a).shape, (answer_target).shape)
                last_output_close, last_output_open, a_close, a_open, decoder,class_pred = model(v, v_p, q, q_token, a,
                                                                                      answer_target)
            elif not args.autoencoder and args.gloria:
                last_output_close, last_output_open, a_close, a_open, gloria_loss,class_pred = model(v, v_p, q, q_token, a,
                                                                                          answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open,class_pred = model(v, v_p, q, q_token, a, answer_target)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)


            #loss
            if preds_close is not None:
                # print("#####", preds_close.shape, a_close.shape) ##### torch.Size([35, 321]) torch.Size([35, 56])
                loss_close = criterion(preds_close.float(), a_close)
            else:
                loss_close = 0

            if preds_open is not None:
                loss_open = criterion(preds_open.float(),a_open)
            else:
                loss_open = 0

            loss = loss_close + loss_open
            if ('RAD' in args.dataset) or ('SLAKE' in args.dataset)or ('FREE' in args.dataset):
                if args.image_classify:
                    loss_img_class = criterion(class_pred['img_class_pred'].float(), image_organ)
                    loss = loss + (loss_img_class * args.img_alpha)
                if args.question_classify:
                    loss_q_class = criterion(class_pred['q_class_pred'].float(), question_type)
                    loss = loss + (loss_q_class * args.q_alpha)

            if args.autoencoder:
                loss_ae = ae_criterion(v[1], decoder)
                loss = loss + (loss_ae * args.ae_alpha)
            if args.gloria:
                loss = loss + (gloria_loss * args.gloria_alpha)



            # loss /= answers.size()[0]
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            #compute the acc for open and close
            if preds_close is not None:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            else:
                batch_close_score = 0
            batch_open_score = compute_score_with_logits(preds_open,a_open.data).sum()
            total_loss += loss.item()
            train_score += batch_close_score + batch_open_score
            number+= q_token.shape[0]

            num_updates += 1
            # if num_updates % int(args.print_interval / update_freq) == 0:
            #     writer.add_scalar('data/trainloss', total_loss / ((num_updates + 1)),
            #                       (epoch * len(train_loader) + i + 1) * args.batch_size)
                    

            # print("#####", num_updates)       
        total_loss /= len(train_loader)
        train_score = 100 * train_score / number
        train_acc_data.append(train_score)
        train_loss_data.append(total_loss)

        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))
        # writer.add_scalar('data/train_epoch_loss', total_loss,epoch)

        sql="INSERT INTO detail (batch_size, epoch,loss,record_id) values(%s, %s, %s, %s)" %  (args.batch_size,epoch+1,total_loss,args.record_id)
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()

        if args.lr_schedule:
            # Evaluation starts
            early_stop_score = None
            decay_metric = None
            do_earl_stop = epoch + 1 >= early_stop_start
            do_lr_decay = epoch + 1 >= decay_metric_start



        # Evaluation
        if eval_loader is not None:
            eval_score,eval_loss = evaluate_classifier(model,question_model, eval_loader, args,logger,epoch)
            test_acc_data.append(eval_score)
            test_loss_data.append(eval_loss)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                # Save the best acc epoch
                model_path = os.path.join(ckpt_path, 'best_model.pth')
                utils.save_model(model_path, model, best_epoch, optim)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))
            # writer.add_scalar('data/val_loss', eval_loss, epoch)
            # writer.add_scalar('data/val_score', eval_score, epoch)
            if args.lr_schedule:
                if early_stop_metric == "eval_acc" and do_earl_stop:
                    early_stop_score = eval_score
        if args.lr_schedule:
            # Record decay_metric (will not be used further if scheduler != ReduceLROnPlateau)
            if do_lr_decay:
                if training_scheduler.decay_on_training_loss:
                    decay_metric = training_loss
                else:
                    decay_metric = early_stop_score

            ret = training_scheduler.eval_step(decay_metric=decay_metric, early_stop_score=early_stop_score)

            if ret["done_training"]:
                f_name1 = args.name + '_loss.png'
                f_name2 = args.name + '_acc.png'
                # x_axis_data = [i for i in range(len(train_loss_data))]
                # draw(x_axis_data, train_loss_data, eval_loss_data, f_name1, 'Loss',args)
                # draw(x_axis_data, train_acc_data, test_acc_data, f_name2, 'Acc',args)

                logger.info("Early stopped reached")
                sys.exit()
            if ret["save_state"]:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                }, os.path.join(ckpt_path, "best_train_model.pth")
                )

    f_name1 = args.name + '_loss.png'
    f_name2 = args.name + '_acc.png'
    cursor.close()
    conn.close()
    # x_axis_data = [i for i in range(len(train_loss_data))]
    # 
    # draw(x_axis_data, train_loss_data, test_loss_data, f_name1, 'Loss',args)
    # draw(x_axis_data, train_acc_data, test_acc_data, f_name2, 'Acc',args)
        
        
        

# Evaluation
def evaluate(model, dataloader, args,logger):
    device = args.device
    score = 0
    total = 0
    open_ended = 0. #'OPEN'
    score_open = 0.

    closed_ended = 0. #'CLOSED'
    score_close = 0.
    model.eval()
    
    with torch.no_grad():
        for i,(v,v_p,image_organ, q,q_token, a, answer_type, question_type, answer_target,qid) in enumerate(dataloader):
            # print("#######", answer_target)
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)
            q_token = q_token.to(device)
            a = a.to(device)

            if args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder,class_pred = model(v, v_p, q, q_token, a, answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open,class_pred = model(v, v_p, q, q_token, a, answer_target)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            batch_close_score = 0.
            batch_open_score = 0.
            ## alter by hxj 
            # if preds_close.shape[0] != 0:
            #     batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            # if preds_open.shape[0] != 0: 
            #     batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()

            if preds_close is not None:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            if preds_open is not None: 
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()
            score += batch_close_score + batch_open_score

            size = q_token.shape[0]
            total += size  # batch number

            if preds_open is not None:
                open_ended += preds_open.shape[0]
                score_open += batch_open_score

            if preds_close is not None:
                closed_ended += preds_close.shape[0]
                score_close += batch_close_score

    score = 100* score / total
    open_score = 100* score_open/ open_ended
    if closed_ended == 0:
        print("no close_score")
        close_score = -999999
    else:
        close_score = 100* score_close/ closed_ended
    print(total, open_ended, closed_ended)
    logger.info('[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%' .format(score,open_score,close_score))
    conn=connect()
    cursor=conn.cursor()
    sql=""
    if args.dataset=='Med-2019':
        sql="UPDATE `record` SET `all`=%f,`status`='%s' where id=%s" % (score,'complete',args.record_id)
    else:
        sql="UPDATE `record` SET closed=%f,open=%f,`all`=%f,`status`='%s' where id=%s" % (close_score,open_score,score,'complete',args.record_id)
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()
    return score

# Load answers
def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]
def get_answer_close(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2close[idx.item()]
def get_answer_open(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2open[idx.item()]

# Evaluation
def evaluate_classifier(model,pretrained_model, dataloader, args,logger,epoch):
    device = args.device
    score = 0
    total = 0
    open_ended = 0. #'OPEN'
    score_open = 0.

    closed_ended = 0. #'CLOSED'
    score_close = 0.
    model.eval()
    pretrained_model.eval()
    total_loss = 0
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    ae_criterion = torch.nn.MSELoss()
    answers=[]
    
    with torch.no_grad():
        # for i,(v,v_p,image_organ, q,q_token, a,answer_type, question_type, answer_target,phrase_type,qid) in enumerate(dataloader):
        for i in iter(dataloader):
            (v, v_p, image_organ, q, q_token, a, answer_type, question_type, answer_target, qid)=i
            # if phrase_type[0] != "freeform":
            #     print(phrase_type)
            #     continue
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)
            q_token = q_token.to(device)
            
            a = a.to(device)
            if ('RAD' in args.dataset) or ('SLAKE' in args.dataset)or ('FREE' in args.dataset):
                image_organ = image_organ.to(device)
            question_type=question_type.to(device)


            if args.autoencoder and args.gloria:
                last_output_close, last_output_open, a_close, a_open, decoder, gloria_loss ,class_pred= model.forward_classify(v, v_p, q, q_token,a, pretrained_model)
            elif args.autoencoder and not args.gloria:#use this YFAN 230103
                last_output_close, last_output_open, a_close, a_open, decoder,class_pred,qid_open,qid_close = model.forward_classify(v, v_p, q, q_token, a,pretrained_model,qid)
            elif not args.autoencoder and args.gloria:
                last_output_close, last_output_open, a_close, a_open, gloria_loss ,class_pred= model.forward_classify(v, v_p, q, q_token, a,pretrained_model)
            else:
                last_output_close, last_output_open, a_close, a_open,class_pred = model.forward_classify(v, v_p, q, q_token, a, pretrained_model)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)
           
            
            
            # q_close_count=preds_close.shape[0]
            # q_open_count = preds_open.shape[0]
            # for index in range(q_close_count):
            #     answer = {}
            #     answer['qid'] = torch.tensor(qid_close[index], device='cpu')
            #     answer['predict'] = get_answer_close(preds_close[index], dataloader)
            #     answer['answer'] = get_answer_close(a_close[index], dataloader)
            #     answers.append(answer)
            # for index in range(q_open_count):
            #     answer = {}
            #     answer['qid'] = torch.tensor(qid_open[index], device='cpu')
            #     answer['predict'] = get_answer_open(preds_open[index], dataloader)
            #     answer['answer'] = get_answer_open(a_open[index], dataloader)
            #     answers.append(answer)
            

            # loss
            # loss_close = criterion(preds_close.float(), a_close)
            # loss_open = criterion(preds_open.float(), a_open)
            # loss = loss_close + loss_open
            # if ('RAD' in args.dataset) or ('SLAKE' in args.dataset)or ('FREE' in args.dataset):
            #     if args.image_classify:
            #         loss_img_class = criterion(class_pred['img_class_pred'].float(), image_organ)
            #         loss = loss + loss_img_class
            #     # if args.question_classify:
            #     #     loss_q_class=criterion(class_pred['q_class_pred'].float(), question_type)
            #     #     loss = loss + loss_q_class
            #
            #
            # if args.autoencoder and args.gloria:
            #     loss_ae = ae_criterion(v[1], decoder)
            #     loss = loss + (loss_ae * args.ae_alpha) + (gloria_loss * args.gloria_alpha)
            # elif args.autoencoder and not args.gloria:
            #     loss_ae = ae_criterion(v[1], decoder)
            #     loss = loss + (loss_ae * args.ae_alpha)
            # elif not args.autoencoder and args.gloria:
            #     loss = loss + (gloria_loss * args.gloria_alpha)

            
            batch_close_score = 0.
            batch_open_score = 0.
            if preds_close.shape[0] != 0:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            if preds_open.shape[0] != 0: 
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()

            score += batch_close_score + batch_open_score

            size = q_token.shape[0]
            total += size  # batch number
            
            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score

    total_loss /= len(dataloader)
   
    score = 100* score / total
    open_score = 100* score_open/ open_ended
    close_score = 100* score_close/ closed_ended
    print(total, open_ended, closed_ended)

    logger.info('[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}% | Val_loss:{:.6f}' .format(score,open_score,close_score,total_loss))
    # import csv
    # csv_name='save_result/'+args.dataset+'/'+str(epoch)+'_'+args.dataset+'.csv'
    # with open(csv_name, "w", newline='') as f:
    #     writer = csv.writer(f)
    #     for answer in answers:
    #         row = [answer['qid'], answer['predict'], answer['answer']]
    #         writer.writerow(row)
    return score,total_loss

