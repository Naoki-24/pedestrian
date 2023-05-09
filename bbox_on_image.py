import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from analysis import get_database


def bbox_on_image(bbox, pid, img_path):
    xtl, ytl, xbr, ybr = bbox
    rect = patches.Rectangle(xy=(xtl,ytl), width=xbr-xtl, height=ybr-ytl,fill=False,ec='#ff0000')
    image = np.array(Image.open(img_path))
    _, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    ax.add_patch(rect)

    plt.savefig(os.path.join('./test_pid_img', pid), bbox_inches='tight',pad_inches = 0)
    plt.close()


if __name__== '__main__':
    database = get_database()
    pid_list = database['pid']
    image_list = database['image']
    bbox_list = database['bbox']

    for pid, image, bbox in zip(pid_list, image_list, bbox_list):
        obs_point = int(len(pid)/2)
        pid = pid[obs_point][0]
        image = image[obs_point]
        bbox = bbox[obs_point]

        bbox_on_image(bbox, pid, image)