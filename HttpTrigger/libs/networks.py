"""
Name: Neural networks file.
Description: This file contains neural network classes.
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import os
import time

import numpy as np
from PIL import Image
from numpy.core.arrayprint import DatetimeFormat
from skimage import io, transform

from ..libs import strings

logger = logging.getLogger(__name__)


def model_detect(model_name):
    print("model detect", model_name)
    """Detects which model to use and returns its object"""
    models_names = strings.MODELS_NAMES
    if model_name in models_names:
        return U2NET(model_name)
    else:
        return False


class U2NET:
    """U^2-Net model interface"""

    def __init__(self, name="u2net"):
      
        print("path is ", os.getcwd())
        import torch
        from torch.autograd import Variable
        from ..libs.u2net import U2NET as U2NET_DEEP
        from ..libs.u2net import U2NETP as U2NETP_DEEP
        self.Variable = Variable
        self.torch = torch
        self.U2NET_DEEP = U2NET_DEEP
        self.U2NETP_DEEP = U2NETP_DEEP

        if name == 'u2net':  # Load model
            logger.debug("Loading a U2NET model (176.6 mb) with better quality but slower processing.")
            net = self.U2NET_DEEP()
        elif name == 'u2netp':
            logger.debug("Loading a U2NETp model (4 mb) with lower quality but fast processing.")
            net = self.U2NETP_DEEP()
        else:
            raise Exception("Unknown u2net model!")
        try:
            if self.torch.cuda.is_available():
                pat = os.path.join("models", name, name + '.pth')
                try:
                    net.load_state_dict(self.torch.load("u2net.pth"))
                    net.cuda()
                except Exception as e:
                    print(e)
            else:
                net.load_state_dict(self.torch.load(os.path.join("models", name, name + '.pth'), map_location="cpu"))
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        net.eval()
        self.__net__ = net  # Define model

    def process_image(self, data, preprocessing=None, postprocessing=None):
        """
        Removes background from image and returns PIL RGBA Image.
        :param data: Path to image or PIL image
        :param preprocessing: Image Pre-Processing Algorithm Class (optional)
        :param postprocessing: Image Post-Processing Algorithm Class (optional)
        :return: PIL RGBA Image. If an error reading the image is detected, returns False.
        """
        from datetime import datetime
        if isinstance(data, str):
            logger.debug("Load image: {}".format(data))
        image, org_image = self.__load_image__(data)  # Load image
        if image is False or org_image is False:
            return False
        dt = datetime.today()  
        seconds = dt.timestamp()
        if preprocessing:  # If an algorithm that preprocesses is specified,
         
            # then this algorithm should immediately remove the background
            image = preprocessing.run(self, image, org_image)
        else:
            print("tdiff", seconds - dt.today().timestamp())
            print("converting")
            image = self.__get_output__(image, org_image)  # If this is not, then just remove the background
        if postprocessing:  # If a postprocessing algorithm is specified, we send it an image without a background
            print("post processing")
            print("tdiff", seconds - dt.today().timestamp())

            image = postprocessing.run(self, image, org_image)
            print("tdiff", seconds - dt.today().timestamp())

        return image

    def __get_output__(self, image, org_image):
        """
        Returns output from a neural network
        :param image: Prepared Image
        :param org_image: Original pil image
        :return: Image without background
        """
        start_time = time.time()  # Time counter
        image = image.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)
        mask, d2, d3, d4, d5, d6, d7 = self.__net__(image)  # Predict mask
        logger.debug("Mask prediction completed")
        # Normalization
        logger.debug("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.__normalize__(mask)
        # Prepare mask
        logger.debug("Prepare mask")
        mask = self.__prepare_mask__(mask, org_image.size)
        # Apply mask to image
        logger.debug("Apply mask to image")
        empty = Image.new("RGBA", org_image.size)
        image = Image.composite(org_image, empty, mask)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __load_image__(self, data):
        """
        Loads an image file for other processing
        :param data: Path to image file or PIL image
        :return: image tensor, original pil image
        """
        image_size = 320  # Size of the input and output image for the model
        if isinstance(data, str):
            try:
                image = io.imread(data)  # Load image if there is a path
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data)
                return False, False
            pil_image = Image.fromarray(image)
        else:
            image = np.array(data)  # Convert PIL image to numpy arr
            pil_image = data
        image = transform.resize(image, (image_size, image_size), mode='constant')  # Resize image
        image = self.__ndrarray2tensor__(image)  # Convert image from numpy arr to tensor
        return image, pil_image

    def __ndrarray2tensor__(self, image: np.ndarray):
        """
        Converts a NumPy array to a tensor
        :param image: Image numpy array
        :return: Image tensor
        """
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image /= np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return self.torch.from_numpy(tmp_img)

    def __normalize__(self, predicted):
        """Normalize the predicted map"""
        ma = self.torch.max(predicted)
        mi = self.torch.min(predicted)
        out = (predicted - mi) / (ma - mi)
        return out

    @staticmethod
    def __prepare_mask__(predict, image_size):
        """Prepares mask"""
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        mask = Image.fromarray(predict_np * 255).convert("L")
        mask = mask.resize(image_size, resample=Image.BILINEAR)
        return mask
