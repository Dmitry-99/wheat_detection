import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_img(image_path, bb, save_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    for i in bb:
        cv2.rectangle(image, (i[0],i[1]), (i[2],i[3]), (0,255,0), thickness=2)
    cv2.imwrite(save_path, image)
    
def plot_loss(
    train_loss,
    val_loss,
    save_path: str = "loss.png",
) -> None:

    #fg, ax = plt.subplots(1, 1, figsize=(19, 5))
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss Curve")
    plt.legend(loc="best")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(save_path)