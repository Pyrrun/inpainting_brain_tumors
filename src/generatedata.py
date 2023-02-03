import cv2
import os
import importlib
import numpy as np
from glob import glob 
from PIL import Image, ImageFilter
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from utils.option import args
from utils.painter import Sketcher



def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image



def demo(args):
    # load images 
    img_list = []
    for ext in ['*.jpg', '*.png']: 
        img_list.extend(glob(os.path.join(args.dir_image, ext)))
    img_list.sort()

    mask_path = glob(os.path.join(args.dir_mask, args.mask_type, '*.png'))

    
    # augmentation 
    mask_trans = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(
            (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location='cpu'))
    model.eval()

    for fn in img_list:
        filename = os.path.basename(fn).split('.')[0]
        orig_img = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (416, 416))
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
        h, w, c = orig_img.shape
        #mask = np.zeros([h, w, 1], np.uint8)
        image_copy = orig_img.copy()
        #mask = cv2.resize(cv2.imread(mask_path[np.random.randint(0, len(mask_path)-1)]), (416, 416))
        mask = Image.open(mask_path[np.random.randint(0, len(mask_path))])
        #mask = mask.convert('L')


        with torch.no_grad():
            #mask_tensor = (ToTensor()(mask)).unsqueeze(0)
            mask_tensor = F.to_tensor(mask_trans(mask)).unsqueeze(0)
            masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
            pred_tensor = model(masked_tensor, mask_tensor)
            comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

            pred_np = postprocess(pred_tensor[0])
            masked_np = postprocess(masked_tensor[0])
            comp_np = postprocess(comp_tensor[0])
            mask_np = postprocess(mask_tensor[0])
            #cv2.imshow('pred_images', mask_np)
            #cv2.waitKey(0)
            #cv2.imshow('pred_images', comp_np)
            #print('inpainting finish!')
            #cv2.waitKey(0)
            cv2.imwrite(os.path.join(args.outputs, f'{filename}_input.png'), comp_np)
            cv2.imwrite(os.path.join(args.outputs, f'{filename}_mask.png'), mask_np)


if __name__ == '__main__':
    demo(args)
