import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
from torch.optim.optimizer import Optimizer
import os

class ImageFolder(Dataset):

    def __init__(self, dir, file_names):

        self.file_names = file_names
        self.dir = dir


    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        img = cv2.imread(os.path.join(self.dir,img_file_name+'.tif'))#Image.open
        if img is None:
           print("Error: Unable to load image")
        #print(img.shape)
        mask = cv2.imread(os.path.join(self.dir.replace("image", "region"),img_file_name+'.tif'),cv2.IMREAD_GRAYSCALE)
        if mask is None:
           print("Error: Unable to load mask")
        #print(img_file_name)
        boundary = cv2.imread(os.path.join(self.dir.replace("image", "boundary"),img_file_name+'.tif'),cv2.IMREAD_GRAYSCALE)
        if boundary is None:
           print("Error: Unable to load boundary")
        dist = cv2.imread(os.path.join(self.dir.replace("image", "distance"),img_file_name+'.tif'),cv2.IMREAD_GRAYSCALE)
        if dist is None:
           print("Error: Unable to load dist")
        
        #Adjusting parameters for color space augmentation
        img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-1, 1),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
        '''
        img, mask, boundary, dist = randomShiftScaleRotate(img, mask, boundary, dist,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
        img, mask, boundary, dist = randomHorizontalFlip(img, mask, boundary, dist)
        img, mask, boundary, dist = randomVerticleFlip(img, mask, boundary, dist)
        img, mask, boundary, dist = randomRotate90(img, mask, boundary, dist)
        '''
        img = image_transform(img)
        mask = mask_transform(mask)
        boundary = boundary_transform(boundary)
        dist = distance_transform(dist)
        
        return img_file_name, img, mask, boundary, dist




def randomHueSaturationValue(image, hue_shift_limit=(-8, 8),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask, boundary, dist,
                           shift_limit=(-0.25, 0.25),  # [-0.25, 1.25]
                           scale_limit=(-0.25, 0.25),  # [-0.25, 1.25]
                           rotate_limit=(-3.14, 3.14),  # [-3.14,3.14]
                           aspect_limit=(-0.25, 0.25),  # [-0.25, 1.25]
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # - 3.14 ~ 3.14
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])  # -0.25, +1.25
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])  # -0.25, +0.25
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        boundary = cv2.warpPerspective(boundary, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        dist = cv2.warpPerspective(dist, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
    return image, mask, boundary, dist


def randomHorizontalFlip(image, mask, boundary, dist, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        boundary = cv2.flip(boundary, 1)
        dist = cv2.flip(dist, 1)

    return image, mask, boundary, dist


def randomVerticleFlip(image, mask, boundary, dist, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        boundary = cv2.flip(boundary, 0)
        dist = cv2.flip(dist, 0)

    return image, mask, boundary, dist


def randomRotate90(image, mask, boundary, dist, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
        boundary = np.rot90(boundary)
        dist = np.rot90(dist)

    return image, mask, boundary, dist

def to_tensor(pic):

    img = torch.from_numpy(pic.copy().transpose((2, 0, 1)))
    return img.float().div(255.0)
    
def image_transform(img):
    
    imagetransform = transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = to_tensor(img)
    img = imagetransform(img)
    return img



def mask_transform(mask):

    mask = mask/255.

    return torch.from_numpy(np.expand_dims(mask, 0)).float()


def boundary_transform(boundary):

    boundary = boundary/255.

    return torch.from_numpy(np.expand_dims(boundary, 0)).float()


def distance_transform(dist):

    dist = dist/255
       
    return torch.from_numpy(np.expand_dims(dist, 0)).float()


class Augment(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = 8 * len(self.dataset)

    def __getitem__(self, idx):
        idx, carry = divmod(idx, 8)
        carry, flipx = divmod(carry, 2)
        transpose, flipy = divmod(carry, 2)

        diry = 2 * flipy - 1
        dirx = 2 * flipx - 1
        base = self.dataset[idx]
        augmented = []
        return tuple(base)#augmented
    def __len__(self):
        return len(self.dataset) * 8

