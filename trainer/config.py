import torch

class CFG:
    mode = 'inference'
    inference_image_path = './data/train/e596e6bb2.jpg'
    save_inference_result = './experiments/pics/result.jpg'
    csv_path = './data/train.csv'
    train_dir = './data/train'
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    num_classes = 2
    pretrained = True
    num_epochs = 20
    weights = './experiments/weights/best_model_15ep.pt' # False or path  #'./experiments/weights/best_model_7ep.pt'
    detection_threshold = 0.7