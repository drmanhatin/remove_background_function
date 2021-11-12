"""
Name: Background remove tool.
Description: This file contains the CLI interface.
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
print("starting main.py imports")
import logging
from time import time
import azure.functions as func
import os
import tempfile
from .libs.networks import U2NET
os.chdir("./HttpTrigger")
 
import logging
import sys
sys.path.append(".")
sys.path.append("./libs")
from .libs import postprocessing as postprocessing
from .libs import preprocessing as preprocessing

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

print("finished main.py imports")
def __work_mode__(path: str):
    """Determines the desired mode of operation"""
    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"


def __save_image_file__(img, file_name, output_path, wmode):
    """
    Saves the PIL image to a file
    :param img: PIL image
    :param file_name: File name
    :param output_path: Output path
    :param wmode: Work mode
    """
    # create output directory if it doesn't exist
    folder = os.path.dirname(output_path)
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    if wmode == "file":
        file_name_out = os.path.basename(output_path)
        if file_name_out == '':
            # Change file extension to png
            file_name = os.path.splitext(file_name)[0] + '.png'
            # Save image
            img.save(os.path.join(output_path, file_name))
        else:
            try:
                # Save image
                img.save(output_path)
            except OSError as e:
                if str(e) == "cannot write mode RGBA as JPEG":
                    raise OSError("Error! "
                                  "Please indicate the correct extension of the final file, for example: .png")
                else:
                    raise e
    else:
        # Change file extension to png
        file_name = os.path.splitext(file_name)[0] + '.png'
        # Save image
        img.save(os.path.join(output_path, file_name))
        


model = U2NET("u2net")
print('setting processing methods')


def process(input_path, output_path):
    """
    Processes the file.
    :param input_path: The path to the image / folder with the images to be processed.
    :param output_path: The path to the save location.
    :param model_name: Model to use.
    :param postprocessing_method_name: Method for image preprocessing
    :param preprocessing_method_name: Method for image post-processing
    """
    
    if input_path is None or output_path is None:
        raise Exception("Bad parameters! Please specify input path and output path.")
 
    
    postprocessing_method = postprocessing.method_detect("aq")
    preprocessing_method = preprocessing.method_detect("aq")
    
    print("method in process is", postprocessing_method, preprocessing_method)
    image = model.process_image(input_path, preprocessing_method, postprocessing_method)
    print("finished processing file")
    __save_image_file__(image, os.path.basename(input_path), output_path, "file")


def main(req: func.HttpRequest) -> func.HttpResponse:

    if(req.get_body().__len__()  < 5): 
        input_path = "tempImagePath.jpg"

    else:  
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg" )
        fp.write(req.get_body())
        fp.close()
        input_path = fp.name

    op = tempfile.NamedTemporaryFile(delete=False, suffix=".png")  
    output_path = op.name
    op.close()
    
    process(input_path, output_path)

    resp = func.HttpResponse(body=open(output_path, 'rb').read(), mimetype="image/png")
    return resp
  
