#  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from flags import DATA_FOLDER

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj

# Local imports
from data import dataset as dset
from models.common import Evaluator
from models.image_extractor import get_image_extractor
from utils.utils import load_args
from utils.config_model import configure_model
from flags import parser
import logging


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

    print('AO weight:{}'.format(args.aow))
    
    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        args=args,
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features,
        train_only=args.train_only,
        subset=args.subset,
        open_world=args.open_world
    )

    valset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='val',
        args=args,
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world
    )

    valoader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8)

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

    log = logging.getLogger("test_logger_{}".format(args.dataset))

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    
    image_extractor.eval()
    for param in image_extractor.parameters():
            param.requires_grad = False

    args.load = ospj(logpath,'ckpt_best_auc.t7')

    checkpoint = torch.load(args.load)
    if image_extractor:
        try:
            image_extractor.load_state_dict(checkpoint['image_extractor'])
            image_extractor.eval()
        except:
            print('No Image extractor in checkpoint')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    if args.model == 'ade':
        evaluator_val = Evaluator(valset)
        best_auc = 0.0
        best_aow = 0.0
        with torch.no_grad():
            for w in range(11):
                args.aow = w * 0.1
                results = test(image_extractor, model,valoader,evaluator_val,args,print_results=False)
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_aow = args.aow
                    print('New best AUC',best_auc)
                    print('A*O weight',best_aow)

        args.aow = best_aow

        if args.open_world:
            log.info("{}_ow best aow: {}".format(args.dataset, best_aow))
        else:
            log.info("{}_cw best aow: {}".format(args.dataset, best_aow))

    evaluator = Evaluator(testset)

    with torch.no_grad():
        test(image_extractor, model, testloader, evaluator, args)
        # evaluator.save_unseen_seen_acc('./curves/' + args.name)


def test(image_extractor, model, testloader, evaluator,  args, print_results=True):
        if image_extractor:
            image_extractor.eval()

        model.eval()

        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
            data = [d.to(device) for d in data]

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
        stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                               topk=args.topk)


        result = ''
        for key in stats:
            result = result + key + '  ' + str(round(stats[key], 4)) + '| '

        result = result + args.name
        if print_results:
            print(f'Results')
            print(result)
        return stats


if __name__ == '__main__':
    main()
