# coding=utf-8
import os
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from trainers.base_model import BaseModel
from nets.net_interface import NetModule
# from trainers.resnet_module import ResnetModel

class ResnetModel(BaseModel):
    def __init__(self, config):
        super(ResnetModel, self).__init__(config)
        self.config = config
        self.interface = NetModule(self.config['model_module_name'], self.config['model_net_name'])
        if 'in_size' in config.keys():
            self.in_size = config['in_size']
        else:
            self.in_size = None
        self.create_model()


    def create_model(self):
        if self.in_size is None:
            self.net = self.interface.create_model(num_classes=self.config['num_classes'])
        else:
            self.net = self.interface.create_model(in_size = self.in_size, num_classes=self.config['num_classes'])
        if torch.cuda.is_available():
            self.net.cuda()

    def load(self):
        # train_mode: 0:from scratch, 1:finetuning, 2:update
        # if not update all parameters:
        # for param in list(self.net.parameters())[:-1]:    # only update parameters of last layer
        #    param.requires_grad = False
        train_mode = self.config['train_mode']
        if train_mode == 'fromscratch':
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net)
            if torch.cuda.is_available():
                self.net.cuda()
            print('from scratch...')

        elif train_mode == 'finetune':
            # self._load()
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net,device_ids=range(torch.cuda.device_count()))
            if torch.cuda.is_available():
                self.net.cuda()
            print('finetuning...')

        elif train_mode == 'inference':
            # self._load()
            print("Infer mode .")
            w_path = os.path.join(self.config['save_path'], self.config['save_name'])
            if os.path.exists(w_path):
                weights = torch.load(w_path)
                # update_weights = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
                # for uw in update_weights:
                #     if uw in weights.keys():
                #         del weights[uw]
                self.net.load_state_dict(weights)
                # torch.save(self.net.state_dict(), "update_policeman_net.pth")
                
            else:
                raise FileNotFoundError("can found weight file : {}".format(w_path))

            print('Loaded weights from file : {}'.format(w_path))

        else:
            ValueError('train_mode is error...')
