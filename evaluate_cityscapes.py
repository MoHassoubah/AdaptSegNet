import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn

from dataset.lidar_dataset.parser import Parser
import yaml
import cv2
from matplotlib import pyplot as plt

from avgmeter import *
from ioueval import *

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = 'C:/lidar_datasets/nuscenes'   #'./data/Cityscapes'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/nuscenes'

IGNORE_LABEL = 255
NUM_CLASSES = 14
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = './snapshots/Kitti2Nuscenes_multi/kitti_90000.pth'    #'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

num_classes = 14

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--kitti", type=bool, default=False,
                        help="Validate for kitti or nuscenes dataset.")
    parser.add_argument(
      '--data_kitti_cfg', '-dck',
      type=str,
      required=False,
      default='dataset/lidar_dataset/config/labels/semantic-kitti.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
      '--data_nuscenes_cfg', '-dcn',
      type=str,
      required=False,
      default='dataset/lidar_dataset/config/labels/semantic-nuscenes.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
      '--arch_cfg', '-ac',
      type=str,
      required=False,
      default='dataset/lidar_dataset/config/arch/sensor_dataset.yaml',
      help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )
    parser.add_argument("--stop-before", type=int, default=95000,
                        help="stop before this itiration")
    return parser.parse_args()
    
def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)

def make_log_img(depth_gt, pred, color_fn, mask, gt):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    
    # depth_gt_img = (cv2.normalize(depth_gt, None, alpha=0, beta=1,
                           # norm_type=cv2.NORM_MINMAX,
                           # dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
            
            
    # out_img = mask[..., None] * cv2.applyColorMap(
        # depth_gt_img, get_mpl_colormap('viridis'))
            
            
            
    # make label prediction
    out_img = color_fn((pred * mask).astype(np.int32))
        
    # out_img = np.concatenate([out_img, pred_color], axis=0)
    
    gt_color = color_fn(gt)
    out_img = np.concatenate([out_img, gt_color], axis=0)
        
    return (out_img).astype(np.uint8)
    
# def fast_hist(a, b, n):
    # k = (a >= 0) & (a < n) #get the indecies of the lables between 0 and 19
    # return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n) #finds the number of repeations of an element in the arrays and returns this number at index = the element


# def per_class_iu(hist):
    # return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def getValidData(): 
    args = get_arguments()
    
    # open arch config file
    try:
      print("Opening arch config file %s" % args.arch_cfg)
      ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    except Exception as e:
      print(e)
      print("Error opening arch yaml file.")
      quit()

      
    # open data config file
    try:
      print("Opening data config file %s" % args.data_nuscenes_cfg)
      DATA_nuscenes = yaml.safe_load(open(args.data_nuscenes_cfg, 'r'))
    except Exception as e:
      print(e)
      print("Error opening data yaml file.")
      quit()
      
    nuscenes_parser = Parser(root=args.data_dir,
                      train_sequences=None,
                      valid_sequences=(700,850),
                      test_sequences=None,
                      labels=DATA_nuscenes["labels"],
                      color_map=DATA_nuscenes["color_map"],
                      learning_map=DATA_nuscenes["learning_map"],
                      learning_map_inv=DATA_nuscenes["learning_map_inv"],
                      sensor=ARCH["dataset_nuscenes"]["sensor"],
                      max_points=ARCH["dataset_nuscenes"]["max_points"],
                      batch_size=ARCH["train"]["batch_size"],
                      workers=ARCH["train"]["workers"],
                      max_iters=None,
                      gt=True,
                      shuffle_train=True,
                      nuscenes_dataset=True)
                      
    return nuscenes_parser
    
    
def main(restore_frm=None,outer_parser=None):
    """Create the model and start the evaluation process."""

    args = get_arguments()

    # if not os.path.exists(args.save + '/gt'):
        # os.makedirs(args.save + '/gt')
        
    # if not os.path.exists(args.save + '/pred'):
        # os.makedirs(args.save + '/pred')

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        if(restore_frm == None):
            saved_state_dict = torch.load(args.restore_from)
        else:
            saved_state_dict = torch.load(restore_frm)
    ### for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    ###
    model.load_state_dict(model_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)

    model.eval()
    
    
    # open arch config file
    try:
      print("Opening arch config file %s" % args.arch_cfg)
      ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    except Exception as e:
      print(e)
      print("Error opening arch yaml file.")
      quit()
    
    
      
    
    if(restore_frm == None):
        if(args.kitti == False):
            # open data config file
            try:
              print("Opening data config file %s" % args.data_nuscenes_cfg)
              DATA_nuscenes = yaml.safe_load(open(args.data_nuscenes_cfg, 'r'))
            except Exception as e:
              print(e)
              print("Error opening data yaml file.")
              quit()
              
            nuscenes_parser = Parser(root=args.data_dir,
                                  train_sequences=None,
                                  valid_sequences=(700,850),
                                  test_sequences=None,
                                  labels=DATA_nuscenes["labels"],
                                  color_map=DATA_nuscenes["color_map"],
                                  learning_map=DATA_nuscenes["learning_map"],
                                  learning_map_inv=DATA_nuscenes["learning_map_inv"],
                                  sensor=ARCH["dataset_nuscenes"]["sensor"],
                                  max_points=ARCH["dataset_nuscenes"]["max_points"],
                                  batch_size=ARCH["train"]["batch_size"],
                                  workers=ARCH["train"]["workers"],
                                  max_iters=None,
                                  gt=True,
                                  shuffle_train=True,
                                  nuscenes_dataset=True)
                                  
            valid_loader  = nuscenes_parser.get_valid_set()
            ignore_classes = [0,7,8,10,16,18,19]
            
            
            img_means = ARCH["dataset_nuscenes"]["sensor"]["img_means"]
            img_stds = ARCH["dataset_nuscenes"]["sensor"]["img_stds"]
            wedith = ARCH["dataset_nuscenes"]["sensor"]["img_prop"]["width"]
            height =ARCH["dataset_nuscenes"]["sensor"]["img_prop"]["height"]
    
            interp_target_rep_row = nn.Upsample(size=(height*2, wedith), mode='nearest')
            
            the_parser = nuscenes_parser
                              
        else:
            # open data config file
            try:
              print("Opening data config file %s" % args.data_kitti_cfg)
              DATA_kitti = yaml.safe_load(open(args.data_kitti_cfg, 'r'))
            except Exception as e:
              print(e)
              print("Error opening data yaml file.")
              quit()
              
            kitti_parser = Parser(root=args.data_dir,
                                  train_sequences=None,
                                  valid_sequences=DATA_kitti["split"]["valid"],
                                  test_sequences=None,
                                  labels=DATA_kitti["labels"],
                                  color_map=DATA_kitti["color_map"],
                                  learning_map=DATA_kitti["learning_map"],
                                  learning_map_inv=DATA_kitti["learning_map_inv"],
                                  sensor=ARCH["dataset_kitti"]["sensor"],
                                  max_points=ARCH["dataset_kitti"]["max_points"],
                                  batch_size=ARCH["train"]["batch_size"],
                                  workers=ARCH["train"]["workers"],
                                  max_iters=None,
                                  gt=True,
                                  shuffle_train=True,
                                  nuscenes_dataset=False)
                                  
            valid_loader  = kitti_parser.get_valid_set()
            ignore_classes = [0]
            
            img_means = ARCH["dataset_kitti"]["sensor"]["img_means"]
            img_stds = ARCH["dataset_kitti"]["sensor"]["img_stds"]
            wedith = ARCH["dataset_kitti"]["sensor"]["img_prop"]["width"]
            height =ARCH["dataset_kitti"]["sensor"]["img_prop"]["height"]
            
            interp_target_rep_row = lambda a : a
            
            the_parser = kitti_parser

    else:
        the_parser = outer_parser
        
    
        valid_loader  = the_parser.get_valid_set()
        ignore_classes = [0]
        
        img_means = ARCH["dataset_nuscenes"]["sensor"]["img_means"]
        img_stds = ARCH["dataset_nuscenes"]["sensor"]["img_stds"]
        wedith = ARCH["dataset_nuscenes"]["sensor"]["img_prop"]["width"]
        height =ARCH["dataset_nuscenes"]["sensor"]["img_prop"]["height"]
    
        interp_target_rep_row = nn.Upsample(size=(height*2, wedith), mode='nearest')
    
    
    # hist = np.zeros((the_parser.get_n_classes(), the_parser.get_n_classes()))

    # testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    # batch_size=1, shuffle=False, pin_memory=True)

    
    interp = nn.Upsample(size=(height, wedith), mode='bilinear', align_corners=True)

    evaluator = iouEval(the_parser.get_n_classes(),device, ignore_classes)
    
    iou = AverageMeter()
    
    with torch.no_grad():
        for index, batch in enumerate(valid_loader):
            if index % 100 == 0:
                print('%d processd' % index)
                                
            in_vol, proj_mask, proj_labels, _, path_seq, name, _, _, _, _, _, _, _, _, _ = batch
            # image, _, name = batch
            in_vol = in_vol.to(device)

            if args.model == 'DeeplabMulti':
                output1, output2 = model(in_vol)
                                                #.data[0]
                output = (interp(output2))#.cpu().numpy()
            elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
                output = model(in_vol)
                                               #.data[0]
                output = (interp(output))#.cpu().numpy()

            # output = output.transpose(1,2,0)
            output = output.squeeze(0).permute(1,2,0)
            
            ########################################################
            proj_labels_iou = proj_labels.cuda(non_blocking=True).long()
            output = output.argmax(dim=2)
            evaluator.addBatch(output, (proj_labels_iou))
            ########################################################
            
            # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                                            #.data[0] that was to use first batch
            ########################################################
            ########################################################
            if index % 100 == 0:
                output = np.asarray(output.cpu().numpy(), dtype=np.uint8)
                
                
                mask_np = proj_mask[0].cpu().numpy()
                gt_np = (proj_labels)[0].cpu().numpy()
                
                # depth_gt_np = in_vol[0,0,:,:].cpu().numpy()
                
                #remove the normalization
                depth_gt_np = None#(depth_gt_np * img_stds[0]) + img_means[0]
            
                
                out = make_log_img(depth_gt_np, output, the_parser.to_color, mask_np, gt_np)
                # print(name)
                save_iter = name[0]
                if restore_frm!=None:
                    tmp_0 = restore_frm.split("/")
                elif args.restore_from[:4] != 'http':
                    tmp_0 = args.restore_from.split("/")
                    
                if restore_frm!=None or args.restore_from[:4] != 'http':
                    tmp_1 = tmp_0[-1]
                    path_tmp1 =tmp_1.split("_")
                    save_iter = path_tmp1[1].split(".")[0]
                name_2_save = os.path.join(args.save, save_iter + '_'+str(index)+'.png')
                cv2.imwrite(name_2_save, out)
            ########################################################
            ########################################################
            
            
            
            
            # if len(gt_np.flatten()) != len(output.flatten()):
                # print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}'.format(len(gt_np.flatten()), len(output.flatten()), name[0]))
                # continue
            
            # hist += fast_hist(gt_np.flatten(), output.flatten(), the_parser.get_n_classes())
            
            # if index > 0 and index % 10 == 0:
                # print('{:d} / {:d}: {:0.2f}'.format(index, the_parser.get_valid_size(), 100*np.mean(per_class_iu(hist))))

            # output_col = colorize_mask(output)
            # output = Image.fromarray(output)

            # name = name[0].split('/')[-1]
            # output.save('%s/%s' % (args.save, name))
            # output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
        
        jaccard, class_jaccard = evaluator.getIoU()
        
        iou.update(jaccard.item(), 1)#in_vol.size(0))    
    # mIoUs = per_class_iu(hist)
    # for ind_class in range(the_parser.get_n_classes()):
        # print('===>' + the_parser.get_xentropy_class_string(ind_class) + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
      # print also classwise
    for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
        i=i, class_str=the_parser.get_xentropy_class_string(i), jacc=round(jacc.item() * 100, 2)))
    print('===> mIoU: ' + str(round(iou.avg * 100, 2)))
    return round(iou.avg * 100, 2)


if __name__ == '__main__':
    main()
