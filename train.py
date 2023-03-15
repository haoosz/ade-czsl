#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

#Local imports
from data import dataset as dset
from models.common import Evaluator
from models.image_extractor import get_image_extractor
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
from torch.optim.lr_scheduler import MultiStepLR
from utils.utils import seed_torch

best_auc = 0
best_hm = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
    seed_torch(args.seed)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        args=args,
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only= args.train_only,
        open_world=args.open_world
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    valset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='val',
        args=args,
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='test',
        args=args,
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    image_extractor.eval()
    
    for param in image_extractor.parameters():
            param.requires_grad = False

    scheduler = MultiStepLR(optimizer, milestones=[300], gamma=0.1)

    evaluator_val =  Evaluator(valset)
    evaluator_test =  Evaluator(testset)

    print(model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)
    
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train(epoch, image_extractor, model, trainloader, optimizer, writer)
    
        if epoch % args.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                test(epoch, image_extractor, model, valloader, evaluator_val, writer, args, logpath, mode='val')
                test(epoch, image_extractor, model, testloader, evaluator_test, writer, args, logpath, mode='test')
        
        scheduler.step()
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)

def train(epoch, image_extractor, model, trainloader, optimizer, writer):
    '''
    Runs training for an epoch
    '''
    model.train() # Let's switch to training

    train_loss = 0.0 
    score_obj, score_attr, score_dis = 0.0, 0.0, 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) if not isinstance(d, tuple) else d for d in data]

        data[0] = image_extractor(data[0])
        data[-1] = image_extractor(data[-1])
        data[-2] = image_extractor(data[-2])
        
        loss, _, scores = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
        score_obj += scores[0].item()
        score_attr += scores[1].item()
        score_dis += scores[2].item()

    train_loss = train_loss/len(trainloader)

    score_obj = score_obj/len(trainloader)
    score_attr = score_attr/len(trainloader)
    score_dis = score_dis/len(trainloader)

    writer.add_scalar('Loss/train_total', train_loss, epoch)
    writer.add_scalar('score/score_obj', score_obj, epoch)
    writer.add_scalar('score/score_attr', score_attr, epoch)
    writer.add_scalar('score/score_dis', score_dis, epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, mode):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    image_extractor.eval()
    model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []

    for _, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) if not isinstance(d, str) else d for d in data]

        data[0] = image_extractor(data[0])

        _, predictions, _ = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(mode + ' ' + key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '
        
    result = result + args.name
    print(f'{mode} Epoch: {epoch}')
    print(result)

    if mode == 'val':
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(epoch)
        if stats['AUC'] > best_auc:
            best_auc = stats['AUC']
            print('New best AUC ', best_auc)
            save_checkpoint('best_auc')

        if stats['best_hm'] > best_hm:
            best_hm = stats['best_hm']
            print('New best HM ', best_hm)
            save_checkpoint('best_hm')

        # Logs
        with open(ospj(logpath, 'logs.csv'), 'a') as f:
            w = csv.DictWriter(f, stats.keys())
            if epoch == 0:
                w.writeheader()
            w.writerow(stats)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)