import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image

from com_ineuron_apparel.com_ineuron_utils.utils import encodeImageIntoBase64

import sys
sys.path.insert(0, 'com_ineuron_apparel/predictor_yolo_detector')

from com_ineuron_apparel.predictor_yolo_detector.models.experimental import attempt_load
from com_ineuron_apparel.predictor_yolo_detector.utils.datasets import LoadStreams, LoadImages
from com_ineuron_apparel.predictor_yolo_detector.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from com_ineuron_apparel.predictor_yolo_detector.utils.torch_utils import select_device, time_sync
from com_ineuron_apparel.predictor_yolo_detector.utils.plots import Annotator, colors, save_one_box

class Detector():
    def __init__(self,filename):
        self.weights = "./com_ineuron_apparel/predictor_yolo_detector/best.pt"
        self.source = "./com_ineuron_apparel/predictor_yolo_detector/inference/images/"
        self.img_size = int(416)
        self.save_dir = "./com_ineuron_apparel/predictor_yolo_detector/inference/output"
        self.device = 'cpu'
        self.augment = True
        self.agnostic_nms = True
        self.conf_thres = float(0.5)
        self.iou_thres = float(0.45)
        self.filename = filename
        self.multi_label = True

    def detect(self):
        out, source, weights,  imgsz = \
            self.save_dir, self.source, self.weights,  self.img_size

        os.makedirs(out,exist_ok=True)
        device = select_device(self.device)

        model = attempt_load(weights,map_location=device)  # change the source code for this project comment out some source code install issue
        imgsz = check_img_size(imgsz,s=model.stride.max())

        dataset = LoadImages(source, img_size=imgsz)
        # print(next(iter(dataset)))
        names = model.names

        # inference
        t0 = time.time()
        for path, img, im0s, vid_cap, s in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img /= 255

            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            pred = model(img,augment=self.augment)[0]

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,multi_label=self.multi_label, 
                                       agnostic=self.agnostic_nms)#  adding multi_label and delect  classes

            
            for det in pred:

                annotator = Annotator(im0s, line_width=4, example="sandeep")

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        labels = f"{names[int(cls)]}:::{conf:.2f}"
                        annotator.box_label(xyxy, labels, color=colors(int(cls), True))
                        
                    
                im = Image.fromarray(im0s)
                im.save("output.jpg")
                time_elapsed = time.time() - t0
                print(f"Time is Taken:::{time_elapsed % 60:.2f}s")
            
        return "OK"

    def detect_action(self):
        with torch.no_grad():
            self.detect()
        bgr_image = cv2.imread("output.jpg")
        im_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('color_img.jpg', im_rgb)
        opencodedbase64 = encodeImageIntoBase64("color_img.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        return result

