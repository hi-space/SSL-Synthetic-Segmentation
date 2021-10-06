import torch.nn as nn
from torch.utils import data, model_zoo
import torch.optim as optim
import torch.nn.functional as F
from model.deeplab_multi import DeeplabMulti
from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
from model.ms_discriminator import MsImageDis
import torch
import torch.nn.init as init
import copy
import numpy as np

import torchvision
import matplotlib
import matplotlib.pyplot as plt
import cv2

matplotlib.use('TkAgg')

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class AD_Trainer(nn.Module):
    def __init__(self, args):
        super(AD_Trainer, self).__init__()
        self.fp16 = args.fp16
        self.class_balance = args.class_balance
        self.often_balance = args.often_balance
        self.num_classes = args.num_classes
        self.class_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.often_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.multi_gpu = args.multi_gpu
        self.only_hard_label = args.only_hard_label
        if args.model == 'DeepLab':
            # self.G = DeeplabMulti(num_classes=args.num_classes, use_se = args.use_se, train_bn = args.train_bn, norm_style = args.norm_style, droprate = args.droprate)
            self.G = Res_Deeplab(num_classes=args.num_classes)

            if args.restore_from[:4] == 'http' :
                saved_state_dict = model_zoo.load_url(args.restore_from)
            else:
                saved_state_dict = torch.load(args.restore_from)
                print("restore from: %s" % (args.restore_from))

            new_params = self.G.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # print i_parts
                if args.restore_from[:4] == 'http' :
                    if i_parts[1] !='fc' and i_parts[1] !='layer5':
                        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                        print('%s is loaded from pre-trained weight.\n'%i_parts[1:])
                else:
                    #new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                    if i_parts[0] =='module':
                        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                        print('%s is loaded from pre-trained weight.\n'%i_parts[1:])
                    else:
                        new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
                        print('%s is loaded from pre-trained weight.\n'%i_parts[0:])
        self.G.load_state_dict(new_params)

        self.D1 = MsImageDis(input_dim = args.num_classes).cuda() 
        self.D2 = MsImageDis(input_dim = args.num_classes).cuda() 
        self.D1.apply(weights_init('gaussian'))
        self.D2.apply(weights_init('gaussian'))

        if self.multi_gpu and args.sync_bn:
            print("using apex synced BN")
            self.G = apex.parallel.convert_syncbn_model(self.G)

        self.gen_opt = optim.SGD(self.G.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        self.dis1_opt = optim.Adam(self.D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

        self.dis2_opt = optim.Adam(self.D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.sm = torch.nn.Softmax(dim = 1)
        self.log_sm = torch.nn.LogSoftmax(dim = 1)
        self.G = self.G.cuda()
        self.D1 = self.D1.cuda()
        self.D2 = self.D2.cuda()
        self.interp = nn.Upsample(size= args.crop_size, mode='bilinear', align_corners=True)
        self.interp_target = nn.Upsample(size= args.crop_size, mode='bilinear', align_corners=True)
        self.lambda_seg = args.lambda_seg
        self.max_value = args.max_value
        self.lambda_me_target = args.lambda_me_target
        self.lambda_kl_target = args.lambda_kl_target
        self.lambda_adv_target1 = args.lambda_adv_target1
        self.lambda_adv_target2 = args.lambda_adv_target2
        self.class_w = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        if args.fp16:
            # Name the FP16_Optimizer instance to replace the existing optimizer
            assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
            self.G, self.gen_opt = amp.initialize(self.G, self.gen_opt, opt_level="O1")
            self.D1, self.dis1_opt = amp.initialize(self.D1, self.dis1_opt, opt_level="O1")
            self.D2, self.dis2_opt = amp.initialize(self.D2, self.dis2_opt, opt_level="O1")

    
    def consistency_loss(self, logits_w, logits_s, target_gt_for_visual, name='ce', T=1.0, p_cutoff=0.0,
                         use_hard_labels=True):
        assert name in ['ce', 'L2']
        logits_w = logits_w.detach()
        if name == 'L2':
            raise NotImplementedError
            # assert logits_w.size() == logits_s.size()
            # return F.mse_loss(logits_s, logits_w, reduction='mean')
        elif name == 'ce':
            pseudo_label = torch.softmax(logits_w, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            # print('max score is: %3f, mean score is: %3f' % (max(max_probs).item(), max_probs.mean().item()))
            mask_binary = max_probs.ge(p_cutoff)
            mask = mask_binary.float()

            if mask.mean().item() == 0:
                acc_selected = 0
            else:
                acc_selected = (target_gt_for_visual[mask_binary] == max_idx[mask_binary]).float().mean().item()

            if use_hard_labels:
                masked_loss = self.Cri_CE_noreduce(logits_s, max_idx) * mask
            else:
                raise NotImplementedError
                # pseudo_label = torch.softmax(logits_w / T, dim=-1)
                # masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
            return masked_loss.mean(), mask.mean(), acc_selected

        else:
            assert Exception('Not Implemented consistency_loss')

    def update_class_criterion(self, labels):
            weight = torch.FloatTensor(self.num_classes).zero_().cuda()
            weight += 1
            count = torch.FloatTensor(self.num_classes).zero_().cuda()
            often = torch.FloatTensor(self.num_classes).zero_().cuda()
            often += 1
            # print(labels.shape)
            n, h, w = labels.shape
            for i in range(self.num_classes):
                count[i] = torch.sum(labels==i)
                if count[i] < 64*64*n: #small objective
                    weight[i] = self.max_value
            if self.often_balance:
                often[count == 0] = self.max_value

            self.often_weight = 0.9 * self.often_weight + 0.1 * often 
            self.class_weight = weight * self.often_weight
            # print(self.class_weight)
            return nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255)

    def update_label(self, labels, prediction):
            criterion = nn.CrossEntropyLoss(weight = self.class_weight, ignore_index=255, reduction = 'none')
            #criterion = self.seg_loss
            loss = criterion(prediction, labels)
            # print('original loss: %f'% self.seg_loss(prediction, labels) )
            #mm = torch.median(loss)
            loss_data = loss.data.cpu().numpy()
            mm = np.percentile(loss_data[:], self.only_hard_label)
            #print(m.data.cpu(), mm)
            labels[loss < mm] = 255
            return labels

    def rand_bbox(self, size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def patch_images(self, images1, images2, labels1, labels2):
        beta = 1.
        lam = np.random.beta(beta, beta)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images1.size(), lam)
        images1[:, :, bbx1:bbx2, bby1:bby2] = images2[:, :, bbx1:bbx2, bby1:bby2]
        labels1[:, bbx1:bbx2, bby1:bby2] = labels2[:, bbx1:bbx2, bby1:bby2]

        return images1, labels1

    def gen_update(self, images, images_t, labels, labels_t, i_iter):
            self.gen_opt.zero_grad()

            # source segmenation loss
            aug_images, aug_labels = self.patch_images(images, images_t, labels, labels_t)

            pred1 = self.G(images)
            pred1 = self.interp(pred1)

            pred2 = self.interp(self.G(aug_images))
            
            if self.class_balance:            
                self.seg_loss = self.update_class_criterion(labels)

            if self.only_hard_label > 0:
                labels1 = self.update_label(labels.clone(), pred1)
                labels2 = self.update_label(aug_labels.clone(), pred2)
                loss_seg1 = self.seg_loss(pred1, labels1)
                loss_seg2 = self.seg_loss(pred2, labels2)
            else:
                loss_seg1 = self.seg_loss(pred1, labels)
                loss_seg2 = self.seg_loss(pred2, labels2)
 
            loss = loss_seg1 + self.lambda_seg * loss_seg1

            # target segmentation loss
            # aug_images_t, aug_labels_t = self.patch_images(images_t, images, labels_t, labels)

            pred_target1 = self.G(images_t)
            pred_target1 = self.interp_target(pred_target1)
            
            if self.multi_gpu:
                loss_adv_target1 = self.D1.module.calc_gen_loss( self.D1, input_fake = F.softmax(pred_target1, dim=1) )
            else:
                loss_adv_target1 = self.D1.calc_gen_loss( self.D1, input_fake = F.softmax(pred_target1, dim=1) )

            loss += self.lambda_adv_target1 * loss_adv_target1

            loss.backward()
            self.gen_opt.step()

            val_loss = self.seg_loss(pred_target1, labels_t)

            return loss_seg1, loss_adv_target1, pred1, pred_target1, val_loss, aug_images
    
    def dis_update(self, pred1, pred_target1):
            self.dis1_opt.zero_grad()

            pred1 = pred1.detach()
            pred_target1 = pred_target1.detach()

            if self.multi_gpu:
                loss_D1, reg1 = self.D1.module.calc_dis_loss( self.D1, input_fake = F.softmax(pred_target1, dim=1), input_real = F.softmax(pred1, dim=1) )
            else:
                loss_D1, reg1 = self.D1.calc_dis_loss( self.D1, input_fake = F.softmax(pred_target1, dim=1), input_real = F.softmax(pred1, dim=1) )

            loss = loss_D1
            
            loss.backward()

            self.dis1_opt.step()

            return loss_D1
