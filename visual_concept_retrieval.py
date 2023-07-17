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
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cos_logits(emb, proto):
    logits = torch.matmul(F.normalize(emb, dim=-1), F.normalize(proto, dim=-1).permute(1,0))
    return logits

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
 
def main():
    # Get arguments and start logging
    parser.add_argument('--image_path', default='clothing/images/green_suit/000002.jpg', help='path of the query image.')
    
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
    args.aow = 1.0

    evaluator = Evaluator(testset)

    with torch.no_grad():
        if image_extractor:
            image_extractor.eval()

        model.eval()

        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        retrieve_visual_concept(feat_extractor, args.image_path, model, trainloader, trainset, args, threshold=None, print_results=True)

def retrieve_visual_concept(feat_extractor, img_path, model, testloader, testset, args, threshold=None, print_results=True):

    model.eval()

    img = load_img(img_path).unsqueeze(0).to(device)

    img = feat_extractor(img)

    attr_i, obj_i = model.get_concept_exclusive(img)
    
    attrfeat_list = []
    objfeat_list = []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):

        # if image_extractor:
        #     data[0] = image_extractor(data[0])

        data[0] = feat_extractor(data[0].to(device))
        attr_feat, obj_feat = model.get_concept_exclusive(data[0])

        attrfeat_list.append(attr_feat)
        objfeat_list.append(obj_feat)

    attr_feat_all = torch.cat(attrfeat_list, 0)
    obj_feat_all = torch.cat(objfeat_list, 0)
    
    attr_match = cos_logits(attr_i, attr_feat_all).squeeze()
    obj_match = cos_logits(obj_i, obj_feat_all).squeeze()
    
    _, attr_ind = torch.sort(attr_match, descending=True)
    _, obj_ind = torch.sort(obj_match, descending=True)
    
    attr_select = attr_ind[:6].cpu().numpy().tolist()
    obj_select = obj_ind[:6].cpu().numpy().tolist()
    
    print('Retrieve {}'.format(img_path))
    print('Same Attribute:')
    for i, idx in enumerate(attr_select):
        print('{}: {}'.format(i,testset.data[idx][0]))
        
    print('Same Object:')
    for i, idx in enumerate(obj_select):
        print('{}: {}'.format(i,testset.data[idx][0]))

if __name__ == '__main__':
    main()

