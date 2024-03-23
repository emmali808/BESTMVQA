import ruamel_yaml as yaml
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='./configs/VQA.yaml') 
    parser.add_argument('--epoch',default=10,type=int) 
    parser.add_argument('--lr',type=float, default=1e-5)
    parser.add_argument('--batch_size',default=8,type=int) 
    args = parser.parse_args()
    config = yaml.load(open(args.file_name, 'r'), Loader=yaml.Loader)
    config['optimizer']['lr']=args.lr
    config['batch_size_train']=args.batch_size
    config['schedular']['epochs']=args.epoch
    with open(args.file_name, 'w') as f:
        data = yaml.dump(config, f)
    print(config)