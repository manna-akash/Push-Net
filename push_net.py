## Inference script for Push-Net

__date__ = "11/09/2022"
__author__ = "Akash Manna"


#Imports
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
from PIL import Image
from colorama import Fore

import numpy as np
import os
import sys
import time
import logging


resultPath = '/home/tomo/dataset/result/'
#sys.path.append("/home/tomo/ws_tomo/src/command/tomo_imaging/src/vision") #NOTE : for rosrun open this line.

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

try:
    import ctypes
    ctypes.cdll.LoadLibrary("libgomp.so.1")

except:
    print("Failed to pre-load libgomp library")


from pushnet_utilities import *
import config as args
from img_utils import * 
        
#from alcon_blister_pushnet import AlconBlister

    


class PushController:
    """
    main function
    """
    def __init__(self, predictor = None, action_evaluator = None):
        self.num_action = args.num_action
        self.bs = args.batch_size
        #self.pred =predictor 
        self.action_evaluator =action_evaluator

        
    def sample_action(self, img, num_actions):
        ''' sample [num_actions] numbers of push action candidates from current img'''
        s = 0.9
        safe_margin = 1.4
        out_margin = 2.0

        ## get indices of end push points inside object mask
        img_inner = cv2.resize(img.copy(), (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img_inner.shape
        img_end = np.zeros((int(H), int(W)))
        img_end[(int(H)-h)//2:(int(H)+h)//2, (int(W)-w)//2:(int(W)+w)//2] = img_inner.copy()
        (inside_y, inside_x) = np.where(img_end.copy()>0)

        ## get indices of start push points outside a safe margin of object
        img_outer1 = cv2.resize(img.copy(), (0,0), fx=safe_margin, fy=safe_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer1.shape
        img_start_safe = np.zeros((int(H), int(W)))
        img_start_safe = img_outer1.copy()[(h-int(H))//2:(h+int(H))//2, (w-int(W))//2:(w+int(W))//2]

        img_outer2 = cv2.resize(img.copy(), (0,0), fx=out_margin, fy=out_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer2.shape
        img_start_out = np.zeros((int(H), int(W)))
        img_start_out = img_outer2.copy()[(h-int(H))//2:(h+int(H))//2, (w-int(W))//2:(w+int(W))//2]

        img_start = img_start_out.copy() - img_start_safe.copy()
        (outside_y, outside_x) = np.where(img_start.copy()>100)

        num_inside = len(inside_x)
        num_outside = len(outside_x)

        actions = []
        for i in range(num_actions):
            start_x = 0
            start_y = 0
            end_x = 0
            end_y = 0
            while True:
                ## sample an inside point
                inside_idx = np.random.choice(num_inside)
                ## sample an outside point
                outside_idx = np.random.choice(num_outside)
                end_x = int(inside_x[inside_idx])
                end_y = int(inside_y[inside_idx])
                start_x = int(outside_x[outside_idx])
                start_y = int(outside_y[outside_idx])

                if start_x < 0 or start_x >= W or start_y < 0 or start_y >= H:
                    print('out of bound')
                    continue
                if img[start_y, start_x] == 0:
                    break
                else:
                    continue

            actions.append(start_x)
            actions.append(start_y)
            actions.append(end_x)
            actions.append(end_y)

        return actions


    def get_best_push(self, Ic, resultPath,SAMPLE_ACTION_VIEW=True, sub=None):
        ''' Input:
                Ic: current image mask
            Output:
                Start and end point cooordinates
        '''
        img_in_curr = Ic.astype(np.uint8)

        _, img_in_curr = cv2.threshold(img_in_curr.copy(), 30, 255, cv2.THRESH_BINARY)


        img_in_curr_ = cv2.imread(args.target_pose).astype(np.uint8)[:,:,0]
        _, img_in_next = cv2.threshold(img_in_curr_.copy(), 30, 255, cv2.THRESH_BINARY)


        ''' Sample actions '''
        actions = self.sample_action(img_in_curr.copy(), self.num_action)

        ''' visualize sampled actions '''
        if SAMPLE_VIS:
            for i in range(len(actions)/4):
                start = [actions[i*4], actions[i*4+1]]
                end = [actions[i*4+2], actions[i*4+3]]
                self.draw_action(img_in_curr.copy(), start, end, single=False, sub = sub)

        ''' Select actions '''
        num_action = len(actions) / 4
        num_action_batch = self.num_action / self.bs
        min_sim_score = 1000

        best_start = None
        best_end = None
        best_sim = None
        best_com = None

        action_batch = []
        hidden = None

        if not METHOD == 'nomem':
            hidden = self.action_evaluator.model.hidden

        action_value_pairs = []
        ##Predicting the best actions from model
        for i in range(int(num_action_batch)):
            ## keep hidden state the same for all action batches during selection
            if not hidden == None:
                self.action_evaluator.model.hidden = hidden
            action = actions[4*i*self.bs: 4*(i+1)*self.bs]
            action_value = self.action_evaluator.evaluate_action(img_in_curr, img_in_next, action)
            action_value_pairs = action_value_pairs + action_value
        
        

        ''' sort action based on sim score '''
        action_value_pairs.sort(key=lambda x : x[1])

        ''' get best push action '''

        pack = action_value_pairs.pop(0)
        best_start = pack[0][0] ## best push starting pixel
        best_end = pack[0][1] ## best push ending pixel

        if SAMPLE_ACTION_VIEW == True:
            #cv2.imshow('Input Image', img_in_curr)
            #cv2.imshow('Target Pose', img_in_next)
            self.draw_action(img_in_curr.copy(), best_start, best_end,resultPath, single=True, sub = sub)


        ''' execute action '''
        ## TODO: do whatever to execute push action (best_start, best_end)

        # ''' update LSTM hidden state '''
        self.action_evaluator.update(best_start, best_end, img_in_curr, img_in_next)

        return best_start, best_end


    def draw_action(self, img, start, end,resultPath, single=True, sub =None):
        """
        Action(s) taken to be considered
        """
        (yy, xx) = np.where(img>0)
        img_3d = np.zeros((int(H), int(W), 3))
        img_3d[yy, xx] = np.array([255,255,255])

        sx = int(start[0])
        sy = int(start[1])
        ex = int(end[0])
        ey = int(end[1])

        cv2.line(img_3d, (sx, sy), (ex, ey), (0,0,255), 3)
        img_3d = img_3d.astype(np.uint8)

        #cv2.imshow('Best Action to be Taken', img_3d)
        cv2.imwrite(resultPath+f"Best_action{sub}.png", img_3d)
        






if __name__=='__main__':

    imagePath =("/home/tomo/ws_tomo/src/command/tomo_imaging/src/vision/pushnet/data/")
    SAMPLE_ACTION_VIEW= True

    from os.path import expanduser
    home = expanduser("~")


    pred= Predictor().initialize() #PushNetInitilization.initilization()

    action_evaluator = evaluation_action(model = pred)

    pred_model =PushController(action_evaluator = action_evaluator)
    models_path = str(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]) + '/src/vision/_models/'
    ONEDRIVE_MODEl_DIR = os.path.join(home, 'tomo_config','model')
        
    alcon_blister_object = AlconBlister(model_path =models_path ,ply_path=ONEDRIVE_MODEl_DIR)

    for image in sorted(os.listdir(imagePath)):
        #pred= Predictor().initialize() #PushNetInitilization.initilization()

        action_evaluator = evaluation_action(model = pred)

        pred_model =PushController(action_evaluator = action_evaluator)

      
        #Making it BW image
        #Ic = cv2.imread(imagePath + image)[:,:,0]
        Ic = Image.open(imagePath + image).convert('RGB')
        alcon_blister_output =alcon_blister_object.inference_yolact(color_frame= Ic)
        Ic = alcon_blister_output[1][:,:,0]

        Ic = cv2.resize(Ic, (128, 106))
        Ic = detectPose(Ic, 0, -7, 7) # -7, 7 : offset values


        prediction_start = time.time()
        push_vec = pred_model.get_best_push(Ic.copy(),SAMPLE_ACTION_VIEW =SAMPLE_ACTION_VIEW)
        prediction_end = time.time()
        predition_time =round(prediction_end - prediction_start, 2)
        rotation_after_action = round(math.degrees(math.atan2(push_vec[1][1]-push_vec[0][1], push_vec[1][0]-push_vec[0][0])),2)


        print(Fore.GREEN+'\033[1m'+ f"\n Start coordinates:  ***** {push_vec[0]} ***** \n the ending coordinates:  ***** {push_vec[1]} ***** .\n The angle made {rotation_after_action}˚\n Model Prediction time {predition_time*1000} ms. \n \n " +'\033[0m')
        


# if __name__=='__main__':

#     imagePath ="/home/tomo/ws_tomo/dataset/blister/blister1/"#("./../input_images/")
#     SAMPLE_ACTION_VIEW= True
#     from os.path import expanduser
#     home = expanduser("~")

#     for image in sorted(os.listdir(imagePath)):

#         pred= PushNetInitilization.initilization()


#         pred_model =PushController(predictor=pred)
#         #Making it BW image
#         Ic = Image.open(imagePath + image).convert('RGB')#.convert("P")#cv2.imread(imagePath + image)#[:,:,0]
        
#         #Ic = detectPose(Ic, 0, -7, 7) # -7, 7 : offset values
#         #print(">>>>>>>>>>>>>>>>>>>>>>>>>",Ic.shape)


#         models_path = str(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]) + '/src/vision/_models/'
#         ONEDRIVE_MODEl_DIR = os.path.join(home, 'tomo_config','model')
        
#         alcon_blister_object = AlconBlister(model_path =models_path ,ply_path=ONEDRIVE_MODEl_DIR)

#         #Ic = np.expand_dims(Ic, axis =0)
        
#         alcon_blister_output =alcon_blister_object.inference_yolact(color_frame= Ic)
#         Ic = alcon_blister_output[1]
        

#         Ic = cv2.resize(Ic, (128, 106))
#         # Ic = torch.from_numpy(Ic)
#         cv2.imwrite("Output_blister.png", Ic)
#         prediction_start = time.time()
#         push_vec = pred_model.get_best_push(Ic,result_path="/home/tomo/ws_tomo/dataset/blister/results/",SAMPLE_ACTION_VIEW =SAMPLE_ACTION_VIEW)
#         prediction_end = time.time()
#         predition_time =round(prediction_end - prediction_start, 2)
#         rotation_after_action = round(math.degrees(math.atan2(push_vec[1][1]-push_vec[0][1], push_vec[1][0]-push_vec[0][0])),2)
#         gc.collect()


#         print(Fore.GREEN+'\033[1m'+ f"\n Start coordinates:  ***** {push_vec[0]} ***** \n the ending coordinates:  ***** {push_vec[1]} ***** .\n The angle made {rotation_after_action}˚\n Model Prediction time {predition_time*1000} ms. \n \n " +'\033[0m')
        





