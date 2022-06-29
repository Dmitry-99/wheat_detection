import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
import shutil
import torch.nn as nn
from skimage import io
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import copy
import random

from trainer.config import CFG
from trainer.utils import *
from trainer.dataset import WheatDataset
from trainer.augmentations import *
from trainer.plots import plot_img

class Trainer():
    def __init__(self):   
        if CFG.mode == 'train':
            print('Dataset creation...')
            self.df = pd.read_csv(CFG.csv_path)
            self.df_new = process_bbox(self.df)

            unq_id = self.df_new['image_id'].unique().tolist()
            train_id = random.sample(unq_id, int(0.8*len(unq_id)))
            train_df = self.df_new[self.df_new['image_id'].isin(train_id)]
            val_df = self.df_new[~self.df_new['image_id'].isin(train_id)]

            self.train_data = WheatDataset(train_df, CFG.train_dir, train_augmentations, mode='train')
            self.val_data = WheatDataset(val_df, CFG.train_dir, valid_augmentations, mode='validation')

            self.train_loader = DataLoader(
                self.train_data,
                batch_size=CFG.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            self.val_loader = DataLoader(
                self.val_data,
                batch_size=CFG.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        print('Model creation...')
        self.model = None
        if CFG.weights:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CFG.num_classes)
            
            state = torch.load(CFG.weights)
            self.model.load_state_dict(state.state_dict())
            self.model.eval()
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=CFG.pretrained)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CFG.num_classes)
        self.model.to(CFG.device)
        
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(params, lr=0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.best_model = None
        
        print('Trainer created...')

        
    def __train__(self):
        val_loss_min = np.inf
        best_epoch = 0
        pred_best_epoch = 0
        
        total_train_loss = []
        total_val_loss = []

        for epoch in range(CFG.num_epochs):
            print(f'\nEpoch: {epoch + 1}')
            start_time = time.time()

            train_loss = []
            val_loss = []

            self.model.train()
            for images, targets, ids in tqdm(self.train_loader): # image_ids

                images = list(image.to(CFG.device) for image in images)
                targets = [{k: v.to(CFG.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets) # 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
                losses = sum(loss for loss in loss_dict.values())

                train_loss.append(losses.item())        
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            epoch_train_loss = np.mean(train_loss)
            total_train_loss.append(epoch_train_loss)
            print(f'Epoch train loss is {epoch_train_loss}')

            # validation
            with torch.no_grad():
                #self.model.eval()
                for images, targets, ids in tqdm(self.val_loader):
                    images = list(image.to(CFG.device) for image in images)
                    targets = [{k: v.to(CFG.device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets) # 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss.append(losses.item())        
                epoch_val_loss = np.mean(val_loss)
                total_val_loss.append(epoch_train_loss)
                print(f'Epoch val loss is {epoch_val_loss}')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            ## TODO: save the model if validation loss has decreased
            if epoch_val_loss <= val_loss_min:
                    print('Train loss decreased ({:.6f} --> {:.6f}).'.format(val_loss_min,epoch_val_loss))
                    # save checkpoint as best model
                    best_model = copy.deepcopy(self.model)
                    pred_best_epoch = best_epoch
                    best_epoch = epoch
                    val_loss_min = epoch_val_loss
                    print(f'Saving best model (epoch {epoch})...')
                    torch.save(self.model.state_dict(), f'experiments/weights/best_model_{best_epoch}ep.pt')
                    if os.path.exists(f'experiments/weights/best_model_{pred_best_epoch}ep.pt'):
                        os.remove(f'experiments/weights/best_model_{pred_best_epoch}ep.pt')

            time_elapsed = time.time() - start_time
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return total_train_loss, total_val_loss
    
    def __inference__(self, image_path):
        image_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_arr /= 255.0
        
        image_id = image_path.split('/')[-1]
        image = test_augmentations(image=image_arr)['image']
        
        #image = torch.unsqueeze(image, dim=0)
        
        image = [image.to(CFG.device)]
        
        detection_threshold = CFG.detection_threshold

        output = self.model(image)[0]

        boxes = output['boxes'].data.cpu().numpy()
        scores = output['scores'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        #return boxes, scores
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }
        print(result['PredictionString'])
        
        plot_img(image_path, boxes, CFG.save_inference_result)
        return result
     
    def run(self, mode, img_path=None):
        if mode == 'train':
            return self.__train__()
        if mode == 'inference':
            return self.__inference__(img_path)