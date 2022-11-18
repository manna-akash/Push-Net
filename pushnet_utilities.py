## Utility functions to help main.py

from push_net_model import *

import numpy as np
import os
import time
import logging
from colorama import Fore
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))


import config as args
from img_utils import * 

''' Dimension of input image'''
W = args.image_resolution["width"]#128.0 ##!!!! Important to make it float to prevent integer division becomes zeros
H = args.image_resolution["height"]#106.0

''' Mode of Goal Configuration Specification'''
#MODE = args.mode["position"] ## uncomment this line if you only care how to re-position an object
#MODE = args.mode["rotation"] ## uncomment this line if you only care how to re-orient an object
MODE = args.mode["reconfigure"]## uncomment this line if care both re-position and re-orient an object

''' Method for comparison '''
METHOD = args.method["with_COM"]#'simcom' ## Original Push-Net
#METHOD = args.method["without_COM"]#'sim' ## Push-Net without estimating COM
#METHOD = args.method["without_memory"]#'nomem' ## Push-Net without LSTM

''' visualization options '''
CURR_VIS = True # display current image
NEXT_VIS = True # display target image
SAMPLE_VIS = False # display all sampled actions
BEST_VIS = True # display the best action
SAMPLE_ACTIONS =True
NUM_ACTION_EXECUTE =5

# logging.basicConfig(format='%(asctime)s %(message)s',filename='pushnet.log', filemode='w', level=logging.INFO)
# print('\033[1m'+f"Input image size {128} 'X' {106}" +'\033[1m')
# print('\033[1m' + "Input image size %.2f MB"%1.1 +'\033[1m')




def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

'''deep neural network predictor'''
class Predictor:
    def __init__(self):
        self.bs = args.batch_size
        model_path = args.model_path#'./model'
        best_model_name = args.arch[METHOD] + '.pth.tar'
        self.model_path = os.path.join(model_path, best_model_name)
        #self.model = self.build_model()
        #calculate time to initialize the model
        start = time.time()
        #self.load_model()
        end =time.time()
        time_elapsed = float(end -start)
        logging.info("Time taken to intilize the model: %.2f ms"% time_elapsed*1000)

    def load_model(self, model):
        try:

            model.load_state_dict(torch.load(self.model_path)['state_dict'])
            if torch.cuda.is_available():
                model.cuda()
            model.eval()
        except FileNotFoundError:
            print(Fore.RED+'\033[1m' +"Model file not found. Check the path variable and filename. EXITING....." + "\033[0m")
            exit()
        return model


    def build_model(self):
        if METHOD == 'simcom':
            return COM_net_sim(self.bs)
        elif METHOD == 'sim':
            return COM_net_sim_only(self.bs)
        elif METHOD == 'nomem':
            return COM_net_nomem(self.bs)


    def initialize(self):
        try: 
            print('\033[1m' +'\033[93m' "Initializing the model...." + '\033[0m')
            initialize_start = time.time()
            model = self.build_model()
            model = self.load_model(model=model)
            initialize_end =time.time()
            print('\033[1m' +'\033[93m' f"Model Intiliazed Sucessfully in ***{round((initialize_end-initialize_start), 2)} sec***: Good Job <3" + '\033[0m')
            return model
        except ValueError:
            print("MODEL IS NOT INITIALIZED PROPERLY. EXITING......")
            SystemExit()


class evaluation_action:
    def __init__(self, model):
        self.model =model
        self.bs = args.batch_size

    def reset_model(self):
        ''' reset the hidden state of LSTM before pushing another new object '''
        self.model.hidden = self.model.init_hidden()

    def update(self, start, end, img_curr, img_goal):
        ''' update LSTM states after an action has been executed'''

        bs = self.bs
        A1 = []
        I1 = []
        Ig = []
        for i in range(bs):
            a1 = [[start[0]/W, start[1]/H, end[0]/W, end[1]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        if METHOD == 'simcom':
            sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

    def evaluate_action(self, img_curr, img_goal, actions):
        ''' calculate the similarity score of actions '''
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []

        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H, actions[4*i+2]/W, actions[4*i+3]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        sim_out = None
        com_out = None

        if METHOD == 'simcom':
            sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)

        sim_np = sim_out.data.cpu().data.numpy()

        if MODE == 'wxy':
            sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
        elif MODE == 'xy':
            sim_sum = np.sum(sim_np[:,1:], 1) # measure (x, y)
        else:
            sim_sum = sim_np[:, 0] # measure (w)

        action_value = []
        for ii in range(len(sim_sum)):
            s = [actions[4 * ii], actions[4 * ii + 1]]
            e = [actions[4 * ii + 2], actions[4 * ii + 3]]
            action_value.append([[s, e], sim_sum[ii]])

        return action_value

