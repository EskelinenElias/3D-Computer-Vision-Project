
import numpy as np
from PIL import Image
# You can import any other files that you use.
# Please also provide pip requirements file for any additional libraries that you might need


def calibrate(imgs):
    """Calibrate the camera using images of a scene.

    Args:
        imgs (list<PIL.Image>): a list of PIL images to be used for calibration

    Returns:
        results of calibration that could be used for finding 3D positions of robot, blocks and target positions.
        They could, for example, contain camera frame, projection matrix, etc.
    """    
    pass

def move_block(blocks, img, calib):
    """Returns the commands to move the specified blocks to their target position.

    Args:
        blocks (list<str>): a list of string values that determine which blocks you should move 
                            and in which order. For example, if blocks = ["red", "green", "blue"] 
                            that means that you need to first move red block. 
                            Your function should at minimum work for the following values of blocks:
                            blocks = ["red"], blocks = ["green"], blocks = ["blue"].
        img (PIL.Image): a PIL image containing the current arrangement of robot and blocks.
        calib : calibration results from function calibrate.

    Returns:
        str: robot commands separated by ";". 
             An example output: "go(20); grab(); turn(90);  go(-10); let_go()"
    """    
    pass