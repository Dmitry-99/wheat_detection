import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset



class WheatDataset(Dataset):
    def __init__(self, data_frame, image_dir, transforms=None, mode='train'):
        super().__init__()
        self.df = data_frame
        self.image_dir = image_dir
        self.images = data_frame['image_id'].unique()
        self.transforms = transforms #get_transforms(phase)
        self.mode = mode
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = self.images[idx] + '.jpg'
        
        image_arr = cv2.imread(os.path.join(self.image_dir,image), cv2.IMREAD_COLOR)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_arr /= 255.0
        
        image_id = str(image.split('.')[0])
        if self.mode == 'test':
            image = self.transforms(image=image_arr)['image']
            return image, image_id
        else:
            point = self.df[self.df['image_id'] == image_id]
            boxes = point[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)

            # there is only one class
            labels = torch.ones((point.shape[0],), dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((point.shape[0],), dtype=torch.int64)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor(idx)
            target['area'] = area
            target['iscrowd'] = iscrowd

            if self.transforms:
                sample = {
                    'image': image_arr,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                }
                sample = self.transforms(**sample)
                image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, 
                                                    zip(*sample['bboxes'])))).permute(1, 0)

            return image, target, image_id