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
from PIL import Image
import torchvision.transforms as transforms
import random
from random import sample
from glob import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(0)
    
def load_img(image):
    # Decide what to output

    # if not self.update_features:
    #     img = self.activations[image]
    # else:
    img = Image.open(ospj(DATA_FOLDER,image)).convert('RGB') #We don't want alpha
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img =transform(img)
    
    return img

def retrieve_txt(feat_extractor, img_path, model, args, threshold=None, print_results=True):
    img_list = []
    for i in img_path:
        img = load_img(i).unsqueeze(0).to(device)
        img_list.append(img)
    img = torch.cat(img_list, 0)
    img = feat_extractor(img)
    
    model.eval()

    _, predictions, _ = model([img, None])

    # attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

    # Gather values as dict of (attr, obj) as key and list of predictions as values
    
    for idx in tqdm(range(args.sample_num)):
        predict = {}
        for k in predictions.keys():
            predict[k] = predictions[k][idx]
            
        pairs_top5 = sorted(predict, key=predict.get, reverse=True)[:5]

        txt_out = '{}: {}-{}, {}-{}, {}-{}, {}-{}, {}-{}\n'.format(img_path[idx], pairs_top5[0][0], pairs_top5[0][1],\
            pairs_top5[1][0], pairs_top5[1][1], pairs_top5[2][0], pairs_top5[2][1], pairs_top5[3][0], pairs_top5[3][1], pairs_top5[4][0], pairs_top5[4][1])
        
        print(txt_out)
        
        if not os.path.exists("./img2txt_retrieval/"):
            os.makedirs("./img2txt_retrieval/")
            print("The new directory img2txt_retrieval is created!")
            
        with open('./img2txt_retrieval/{}_unseen_ow.txt'.format(args.dataset), 'a') as f:
            f.write(txt_out)  
               
def main():
    # Get arguments and start logging
    parser.add_argument('--sample_num', default=100, type=int, help='sample number (-1: all samples)')
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        args = args,
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features,
        train_only=args.train_only,
        subset=args.subset,
        open_world=args.open_world
    )


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
    args.aow = 1.0

    with torch.no_grad():
        if image_extractor:
            image_extractor.eval()

        model.eval()

        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        # for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Retrieving'):
        #     data = [d.to(device) for d in data]
        path = os.path.join(DATA_FOLDER,args.data_dir)
        root = os.path.join(path,'images')
        train_pairs_dir = os.path.join(path,'compositional-split-natural/train_pairs.txt')
        with open(train_pairs_dir, 'r') as f:
            train_pairs = f.read().strip().split('\n')
        
        files_before = glob(os.path.join(root, '**', '*.jpg'), recursive=True)
        files_all = []
        for current in files_before:
            parts = current.split('/')
            if "cgqa" in root:
                files_all.append(parts[-1])
            else:
                if parts[-2] not in train_pairs:
                    files_all.append(os.path.join(parts[-2],parts[-1]))
        
        assert args.sample_num != 0, "sample number is zero!"
        if args.sample_num > 0:
            path_list = sample(files_all, args.sample_num)
        else:
            path_list = files_all
        
        img_path = ['{}/images/'.format(args.data_dir) + p for p in path_list]

        retrieve_txt(feat_extractor, img_path, model, args, threshold=None, print_results=True)

if __name__ == '__main__':
    main()
