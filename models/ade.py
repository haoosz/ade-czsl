import torch
import torch.nn as nn
import torch.nn.functional as F
import os, random
import numpy as np
from PIL import Image
import colorsys

from .word_embedding import load_word_embeddings
from .common import MLP
from itertools import product
from .emd_utils import emd_inference_opencv_test, emd_inference_qpth
import matplotlib.pyplot as plt  

from .multi_head_attention import CrossAttention

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to('cpu')

class ADE(nn.Module):

    def __init__(self, dset, args):
        super(ADE, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
        
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        def custom_utzappos(vocab):
            vocab = [v.lower() for v in vocab]
            custom_map = {
                'faux.fur': 'fake_fur',
                'faux.leather': 'fake_leather',
                'full.grain.leather': 'thick_leather',
                'hair.calf': 'hair_leather',
                'patent.leather': 'shiny_leather',
                'boots.ankle': 'ankle_boots',
                'boots.knee.high': 'knee_high_boots',
                'boots.mid-calf': 'midcalf_boots',
                'shoes.boat.shoes': 'boat_shoes',
                'shoes.clogs.and.mules': 'clogs_shoes',
                'shoes.flats': 'flats_shoes',
                'shoes.heels': 'heels',
                'shoes.loafers': 'loafers',
                'shoes.oxfords': 'oxford_shoes',
                'shoes.sneakers.and.athletic.shoes': 'sneakers',
                'traffic_light': 'traffic_light',
                'trash_can': 'trashcan',
                'dry-erase_board' : 'dry_erase_board',
                'black_and_white' : 'black_white',
                'eiffel_tower' : 'tower',
                'nubuck' : 'grainy_leather',
            }
            vocab_new = []
            for v in vocab:
                if v in custom_map:
                    v = custom_map[v]
                vocab_new.append(v)
            return vocab_new

        def get_txt(relative_pairs, fit_utzps=False):
            pair_txt = []
            for _, pair in enumerate(relative_pairs):
                attr, obj = pair
                if fit_utzps:
                    attr = custom_utzappos([attr])[0]
                    obj = custom_utzappos([obj])[0]

                pair_txt.append(attr + ' ' + obj)
            return pair_txt

        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)

        self.scale = self.args.cosine_scale

        self.attr_txt = self.dset.attrs
        self.obj_txt = self.dset.objs
        self.pair_txt = get_txt(self.dset.pairs, fit_utzps=args.dataset == 'utzappos')

        if args.dataset == 'utzappos':
            self.attr_txt = custom_utzappos(self.attr_txt)
            self.obj_txt = custom_utzappos(self.obj_txt)

        if dset.open_world:
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

            self.activated = False
                
        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        
        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                dropout=self.args.dropout,
                                norm=self.args.norm, layers=layers)

        self.image_embedder_obj = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                dropout=self.args.dropout,
                                norm=self.args.norm, layers=layers)

        self.image_embedder_attr = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                dropout=self.args.dropout,
                                norm=self.args.norm, layers=layers)

        input_dim = args.emb_dim

        if "dino" in args.image_extractor:
            self.cross_attn = CrossAttention(768, 12)
            self.cross_attn_attr = CrossAttention(768, 12)
            self.cross_attn_obj = CrossAttention(768, 12)
        else:
            self.cross_attn = CrossAttention(512, 8)
            self.cross_attn_attr = CrossAttention(512, 8)
            self.cross_attn_obj = CrossAttention(512, 8)


        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        
        # init with word embeddings
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(input_dim * 2, args.emb_dim)

    def select_simi_feat(self, x, x_bar): # b, c, hw

        def emd_flow(x, x_bar):
            thred = 0.8

            x_norm = F.normalize(x, dim=1)
            x_bar_norm = F.normalize(x_bar, dim=1)
            cos_sim_mat = torch.bmm(x_norm.transpose(1,2), x_bar_norm) # b, hw, hw
            w = torch.bmm(x_norm.transpose(1,2), F.normalize(torch.max(x_bar_norm, 2, keepdim=True)[0], dim=1)).squeeze() # (b, hw, c)*(b, c, 1) = (b, hw, (1))
            w_bar = torch.bmm(x_bar_norm.transpose(1,2), F.normalize(torch.max(x_norm, 2, keepdim=True)[0], dim=1)).squeeze()

            _, flow = emd_inference_opencv_test(1-cos_sim_mat, w, w_bar) # b, hw. hw
            # _, flow = emd_inference_qpth(1-cos_sim_mat, w, w_bar)
            return flow > thred

        c = x.shape[1]
        flowx = emd_flow(x, x_bar) # b, hw, hw
        mask = flowx.sum(2, keepdim=True).repeat(1,1,c).transpose(1,2) # b, c, hw
        mask_bar = flowx.sum(1, keepdim=True).repeat(1,c,1) # b, c, hw

        return mask, mask_bar

    def select_low_entropy(self, x, x_bar, emb): # x --> b, c, hw; emb --> c, n
        def compute_entropy(p): # batch, hw, n_logits
            p = F.softmax(p, dim=-1)
            logp = torch.log(p)
            entropy = torch.sum(-p*logp, dim=-1, keepdim=True) # batch, hw
            return entropy

        c = x.shape[1]
        x = F.normalize(x.transpose(1,2), dim=-1)
        x_bar = F.normalize(x_bar.transpose(1,2), dim=-1)
        
        entropy = compute_entropy(torch.matmul(x, emb)) # b, hw
        entropy_bar = compute_entropy(torch.matmul(x_bar, emb)) # b, hw

        mask_e = F.softmax(-entropy, dim=1).repeat(1,1,c)
        mask_e_bar = F.softmax(-entropy_bar, dim=1).repeat(1,1,c)

        return mask_e.transpose(1,2), mask_e_bar.transpose(1,2)

    def _setup_word_composer(self):

        # Composer conditioned on object.
        self.object_code = nn.Sequential(
            nn.Linear(self.args.emb_dim, 300),
            nn.ReLU(True)
        )
        self.attribute_code = nn.Sequential(
            nn.Linear(self.args.emb_dim, 300),
            nn.ReLU(True)
        )
        self.attribute_code_fc = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
        )
        self.compose = MLP(
            self.args.emb_dim + 300, self.args.emb_dim, 2, dropout = 0.1, norm = False, layers=[300]
        )

    def compose_word_embeddings(self):

        attrs, objs = self.attr_embedder(self.train_attrs), self.obj_embedder(self.train_objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        # output = F.normalize(output, dim=1)
        return output

    def compose_img_embeddings(self, img_obj, img_attr):
        obj = self.object_code(img_obj)
        attr = self.attribute_code(img_attr) # [n_pairs, 300].
        attr = self.attribute_code_fc(obj * attr)
        # attribute_c = object_c * attribute_c
        concept_emb = torch.cat((obj, attr), dim=-1) # [n_pairs, word_dim + 1024].
        concept_emb = self.compose(concept_emb) # [n_pairs, emb_dim].

        return F.normalize(concept_emb, dim=-1)

    def cos_logits(self, emb, proto):
        logits = torch.matmul(F.normalize(emb, dim=-1), F.normalize(proto, dim=-1).permute(1,0))
        return logits

    def compute_EMD_loss(self, x, y, return_flow=False): # input: B, n_tokens, dim
        def avg_pooling(attn):
            b, n_token, _ = attn.shape
            n_patch = n_token - 1
            h = int(n_patch ** 0.5)
            attn1 = attn[:, :, 1:]
            cls1 = attn[:, : , 0].unsqueeze(-1)

            pos1 = attn1.reshape(b, -1, h, h)
            pos1 = F.avg_pool2d(pos1, kernel_size=2, stride=2)
            attn_new = torch.cat([cls1, pos1.reshape(b, n_token, -1)], dim=-1).transpose(1,2)

            attn2 = attn_new[:, :, 1:]
            cls2 = attn_new[:, :, 0].unsqueeze(-1)

            pos2 = attn2.reshape(b, -1, h, h)
            pos2 = F.avg_pool2d(pos2, kernel_size=2, stride=2)
            attn_new = torch.cat([cls2, pos2.reshape(b, -1, int((h/2)**2))], dim=-1).transpose(1,2)
            
            return attn_new # b, dim, tokens

        if x.shape[-1]>100:
            x = avg_pooling(x)
            y = avg_pooling(y)

        w_y = x[:, 0, 1:]
        w_x = y[:, 0, 1:]
        w_y = w_y / w_y.sum(-1).unsqueeze(-1)
        w_x = w_x / w_x.sum(-1).unsqueeze(-1)
        
        mat = (x[:, 1:, 1:] + y[:, 1:, 1:].transpose(1,2)) / 2.

        _, flow = emd_inference_opencv_test(1-mat, w_x, w_y) # b, hw. hw
        if return_flow:
            return flow
        score = (flow * mat).sum(-1).sum(-1)
        return score.mean()

    def val_forward(self, x):
        img = x[0]

        img_attn, attn_comp = self.cross_attn(q=img, k=img, return_attention=True)
        img_attn_attr, attn_attr = self.cross_attn_attr(q=img, k=img, return_attention=True)
        img_attn_obj, attn_obj = self.cross_attn_obj(q=img, k=img, return_attention=True)
        
        concept = self.compose_word_embeddings()
        
        img_proj = self.image_embedder(img_attn[:,0,:])
        img_proj_obj = self.image_embedder_obj(img_attn_obj[:,0,:])
        img_proj_attr = self.image_embedder_attr(img_attn_attr[:,0,:])

        pair_pred = self.cos_logits(img_proj, concept)

        obj_pred = self.cos_logits(img_proj_obj, self.obj_embedder.weight)
        attr_pred = self.cos_logits(img_proj_attr, self.attr_embedder.weight)

        scores = {}

        for _, pair in enumerate(self.dset.pairs):
            attr, obj = pair
            
            scores[pair] = pair_pred[:,self.dset.pair2idx[pair]] + attr_pred[:, self.dset.attr2idx[attr]] * obj_pred[:, self.dset.obj2idx[obj]] * self.args.aow

        return None, scores, [attn_comp, attn_attr, attn_obj]

    def train_forward(self, x):
        img, attrs, objs, pairs, img_same_attr, img_same_obj, image_path, image_attr_path, image_obj_path = x[0], x[1], x[2], x[3], x[-2], x[-1], x[-5], x[-4], x[-3]

        img_attn_obj, attn_obj = self.cross_attn_obj(q=img, k=img_same_obj, return_attention=True)
        img_attn_obj_bar, attn_obj_bar = self.cross_attn_obj(q=img_same_obj, k=img, return_attention=True)

        _, attn_diff_obj = self.cross_attn_obj(q=img, k=img_same_attr, return_attention=True)
        _, attn_diff_obj_bar = self.cross_attn_obj(q=img_same_attr, k=img, return_attention=True)

        img_attn_attr, attn_attr = self.cross_attn_attr(q=img, k=img_same_attr, return_attention=True)
        img_attn_attr_bar, attn_attr_bar = self.cross_attn_attr(q=img_same_attr, k=img, return_attention=True)
        
        _, attn_diff_attr = self.cross_attn_attr(q=img, k=img_same_obj, return_attention=True)
        _, attn_diff_attr_bar = self.cross_attn_attr(q=img_same_obj, k=img, return_attention=True)

        attn_obj = attn_obj.sum(1)
        attn_obj_bar = attn_obj_bar.sum(1)
        
        attn_attr = attn_attr.sum(1)
        attn_attr_bar = attn_attr_bar.sum(1)

        attn_diff_obj = attn_diff_obj.sum(1)
        attn_diff_obj_bar = attn_diff_obj_bar.sum(1)
        
        attn_diff_attr = attn_diff_attr.sum(1)
        attn_diff_attr_bar = attn_diff_attr_bar.sum(1)

        score_obj = self.compute_EMD_loss(attn_obj, attn_obj_bar)
        score_attr = self.compute_EMD_loss(attn_attr, attn_attr_bar)
        score_diff_attr = self.compute_EMD_loss(attn_diff_attr, attn_diff_attr_bar)
        score_diff_obj = self.compute_EMD_loss(attn_diff_obj, attn_diff_obj_bar)
        score_dis = score_diff_attr + score_diff_obj

        loss_emd = score_dis - (score_attr + score_obj)

        img_attn = self.cross_attn(q=img, k=img)

        img_attn_obj = self.image_embedder_obj(img_attn_obj[:,0,:])
        img_attn_obj_bar = self.image_embedder_obj(img_attn_obj_bar[:,0,:])
        img_attn_attr = self.image_embedder_attr(img_attn_attr[:,0,:])
        img_attn_attr_bar = self.image_embedder_attr(img_attn_attr_bar[:,0,:])
        img_attn = self.image_embedder(img_attn[:,0,:])

        concept = self.compose_word_embeddings()

        logit_attr = self.cos_logits(img_attn_attr, self.attr_embedder.weight)
        logit_attr_bar = self.cos_logits(img_attn_attr_bar, self.attr_embedder.weight)
        logit_obj = self.cos_logits(img_attn_obj, self.obj_embedder.weight)
        logit_obj_bar = self.cos_logits(img_attn_obj_bar, self.obj_embedder.weight)

        logit_comp = self.cos_logits(img_attn, concept)

        loss_comp = F.cross_entropy(self.scale * logit_comp, pairs)
        
        loss_obj = F.cross_entropy(self.scale * logit_obj, objs)
        loss_attr = F.cross_entropy(self.scale * logit_attr, attrs)

        loss_obj_bar = F.cross_entropy(self.scale * logit_obj_bar, objs)
        loss_attr_bar = F.cross_entropy(self.scale * logit_attr_bar, attrs)
        
        return (loss_comp + loss_obj + loss_attr + loss_obj_bar + loss_attr_bar + loss_emd).mean(), None, (score_obj, score_attr, score_dis)
    
    def get_concept_exclusive(self, x):
        img = x

        img_attn, attn_comp = self.cross_attn(q=img, k=img, return_attention=True)
        img_attn_attr, attn_attr = self.cross_attn_attr(q=img, k=img, return_attention=True)
        img_attn_obj, attn_obj = self.cross_attn_obj(q=img, k=img, return_attention=True)
        
        img_proj = self.image_embedder(img_attn[:,0,:])
        img_proj_obj = self.image_embedder_obj(img_attn_obj[:,0,:])
        img_proj_attr = self.image_embedder_attr(img_attn_attr[:,0,:])

        return img_proj_attr, img_proj_obj
    
    def forward(self, x):
        if self.training:
            loss, pred, scores = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred, scores = self.val_forward(x)
        return loss, pred, scores