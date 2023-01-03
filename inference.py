# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from importlib import import_module
from utils.config import process_config
from trainers.faceq_module import FaceQModel
import glob
from data_loader.data_processor import DataProcessor
import shutil
__all__ = [cv2]


def pytorch2json(model:torch.nn.Module, input:torch.Tensor):
    from pytorch2json import auto2json
    #model.set_trans_model()
    auto2json.run(model, input)
    exit(0)


def pytorch_jit(net:torch.nn.Module, input:torch.Tensor, save_path:str):
    #net.set_trans_model()
    if 'cuda' in str(input.device):
        input = input.to(torch.device('cpu'))
        net = net.cpu()
    output = net(input)
    traced_cell = torch.jit.trace(net, (input))
    torch.jit.save(traced_cell, save_path)
    exit(0)


class TagPytorchInference(object):

    def __init__(self, **kwargs):
        _input_size = kwargs.get('input_size',128)
        self.input_size = (_input_size, _input_size)
        self.gpu_index = kwargs.get('gpu_index', '0')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index
        assert  os.path.exists(kwargs.get('cfg'))
        self.config = process_config(kwargs.get('cfg'))
        self.module = FaceQModel(self.config)
        self.dataprocessor = DataProcessor(self.config)
        
        # self._load(**kwargs)
        self.module.load()
        self.net = self.module.net
        self.net.eval()


        self.transforms = transforms.ToTensor()
        if torch.cuda.is_available():
            self.net.cuda()

    def close(self):
        torch.cuda.empty_cache()

    def prepare_input(self, filename):
        image = self.dataprocessor.image_loader(filename)
        image = self.dataprocessor.image_squre_resize(image)
        input = self.dataprocessor.input_norm(image)
        input = self.transforms(input)
        return input, image.copy()[:, :, ::-1]

    def predict(self, fpath):
        assert os.path.exists(fpath)
        input, v_image = self.prepare_input(fpath)
        input = torch.unsqueeze(input, 0)
        if torch.cuda.is_available():
            input = input.cuda()
        logit = self.net(Variable(input))
        infer = F.softmax(logit, 1)
        infer = infer.argmax(1).cpu().item()
        return infer, v_image

if __name__ == "__main__":
    # # python3 inference.py --image test.jpg --module inception_resnet_v2_module --net inception_resnet_v2 --model model.pth
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', "--image", type=str, help='Assign the image path.', default=None)
    parser.add_argument('-idir', "--image_dir", type=str, help='Assign the image directory.', default=r'./data_example/mnist_images')
    parser.add_argument('-cfg', "--config", type=str, help='Assign the config file.', default='configs/mnist_infer_config.json')
    parser.add_argument('-net', "--net", type=str, help='Assign the net name.', default='GeM_ResNet_AVP')
    parser.add_argument('-model', "--model", type=str, help='Assign the net name.', default='GeM_ResNet_module')
    parser.add_argument('-cls', "--cls", type=int, help='Assign the classes number.', default=10)
    parser.add_argument('-size', "--size", type=int, help='Assign the input size.', default=128)
    #是否转json
    parser.add_argument('--pytorch2json', default=False, type=bool, help='convert from pytorch to json or not.')
    #是否转jit
    parser.add_argument('--pytorch2jit', default=False, type=bool, help='convert from pytorch to jit(pt) or not.')
    #是否转onnx
    parser.add_argument('--pytorch2onnx', default=False, type=bool, help='convert from pytorch to onnx or not.')
    args = parser.parse_args()

    # if args.image is None or args.module is None or args.net is None or args.model is None\
    #         or args.size is None or args.cls is None:
    #     raise TypeError('input error')
    # if not os.path.exists(args.model):
    #     raise TypeError('cannot find file of model')
    # if not os.path.exists(args.image):
    #     raise TypeError('cannot find file of image')

    print('test:')
    filename = args.image
    file_dir = args.image_dir
    # module_name = args.module
    net_name = args.net
    model_name = args.model
    input_size = args.size
    num_classes = args.cls
    pro_config = args.config
    labels = ["playphone","work"]
    tagInfer = TagPytorchInference(cfg=pro_config, net_name=net_name,
                                   num_classes=num_classes, model_name=model_name,
                                   input_size=input_size)

    #转json
    if args.pytorch2json:
        net = tagInfer.module.net
        net.eval()
        input_randn = torch.randn([1, 3, input_size, input_size])
        input_randn.requires_grad = False
        #转json和weights
        pytorch2json(net.cpu(), input_randn)

    #转模型操作，转pt文件
    if args.pytorch2jit:
        net = tagInfer.module.net
        net.eval()
        input_randn = torch.randn([1, 3, input_size, input_size])
        input_randn.requires_grad = False
        # save jit model
        pytorch_jit(net.cpu(), input_randn, "./mnist_TensorRT.pt")

    #转模型操作，转onnx文件
    if args.pytorch2onnx:
        net = tagInfer.module.net
        net.eval()
        input_randn = torch.randn([1, 3, input_size, input_size])
        input_randn.requires_grad = False
        # 转onnx(这个要在GPU上转)
        torch.onnx.export(net, input_randn.cuda(), "./mnist_TensorRT.onnx")
    
    if filename is not None:
        image = cv2.imread(filename)
        if image is None:
            raise TypeError('image data is none')

        result = tagInfer.run(image)
        print(result)
    if file_dir is not None:
        assert os.path.exists(file_dir)

        im_paths = glob.glob(os.path.join(file_dir, "*.*"))
        im_paths = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), im_paths))
        for i, im_path in enumerate(im_paths):
            print("Predicting No: {}".format(i))
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if img is not None:
                result, show_img = tagInfer.predict(im_path)
                # if result == 0:
                #     shutil.copy(im_path, "temp2/")

                #show_img = cv2.putText(cv2.UMat(show_img).get(), str(labels[int(result)]), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0), 1)

                print("image_name:", os.path.basename(im_path), "; result:" , result)

                #cv2.imshow('police', show_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                #将预测为玩手机的图片另存在一个文件夹（result）中
                img_copy = img.copy()
                name = os.path.basename(im_path)
                path = os.path.join("./data_example/mnist_result", name)
                if int(result) == 0:
                    cv2.imwrite(path, img_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    continue
    print('done!')
