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



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    parser.add_argument('--text_prompt', default='squatting catcher', type=str, help='Give a text prompt for retrieval.')
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

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
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=512,
        shuffle=False,
        num_workers=args.workers)
    
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
        batch_size=512,
        shuffle=False,
        num_workers=args.workers)

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor
    
    feat_extractor = get_image_extractor(arch = args.image_extractor).eval()
    feat_extractor = feat_extractor.to(device)
    for param in feat_extractor.parameters():
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

    threshold = None

    args.aow = 0.1

    texta, texto = args.text_prompt.strip().split(' ')
    pair = (texta, texto)
    with torch.no_grad():
        retrieve_img(image_extractor, pair, feat_extractor, model, testloader, testset, args, threshold)


def retrieve_img(image_extractor, pair, feat_extractor, model, testloader, testset, args, threshold=None, print_results=True):
    if image_extractor:
        image_extractor.eval()

    model.eval()

    all_pred = []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):

        # if image_extractor:
        #     data[0] = image_extractor(data[0])

        data[0] = feat_extractor(data[0].to(device))
        _, predictions, _ = model(data)

        all_pred.append(predictions[pair])

    all_pred = torch.cat(all_pred, 0)
    _, ind = torch.sort(all_pred, descending=True)
    
    select = ind[:5].cpu().numpy().tolist()
    
    print('Retrieve {}'.format(pair))
    for i, idx in enumerate(select):
        print('{}: {}'.format(i,testset.data[idx][0]))

if __name__ == '__main__':
    main()

