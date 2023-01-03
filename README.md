
# 工程简介
本项目用于训练和测试多分类模型：
- 项目名称：image_classification_pytorch
- 训练框架：pytorch
- 基础镜像：yolo_yolox
- 基础镜像获取方式：docker pull fanacio/yolo_yolox:v0

## 1. 安装依赖库
本人选择的镜像是提前制作好的yolo_yolox训练镜像，已经包含了很多依赖库了，且将没有包含的依赖库提炼在了requirements_fanyichao.txt文件中，所以本人安装依赖库的方法是直接进入yolo_yolox容器，执行如下命令即可：
```bash
pip install -i https://pypi.douban.com/simple -r requirements_fanyichao.txt
```
也可以选择其他源进行安装，如果选择使用dockerhub中pull下的仅包含python的基础镜像，那么直接执行如下命令即可安装所有依赖库：
```bash
pip install -i https://pypi.douban.com/simple -r requirements.txt
```

## 2. 测试此工程
在依赖库安装完成后需要先测试工程是否存在问题，本人保存了一个玩手机检测的模型作为测试模型，执行如下代码即可完成测试：
```bash
python inference.py
```
测试图片存放在image_classification_pytorch/data_example/images目录下，测试结果存放在image_classification_pytorch/data_example/result目录下，且在终端打印出来。

## 3. 训练模型
### 3.1 训练前准备之数据集的准备
本次训练以MNIST数据集为例，以下是关于数据集的获取方法：
- 操作平台：linux系统docker
- 操作步骤：
    - step1:新建获取数据集的python工程
    ```
    工作目录新建一个mnist_dataset的文件夹，在此文件夹下创建get_dataset.py文件，并写入以下内容：
    ```
    ```python
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import os
    from PIL import Image
    import numpy as np

    MNIST_data_folder = 'MNIST_data_folder'
    mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=False) #MNIST_data_folder是数据集的目录
    # imgs, labels = mnist.test.images, mnist.test.labels #生成测试集图片
    imgs, labels = mnist.validation.images, mnist.validation.labels  #生成验证集图片
    # imgs, labels = mnist.train.images, mnist.train.labels #生成训练集图片
    for i in range(10):
        if not os.path.exists(str(i)):
            os.makedirs(str(i))
    cnt = [0 for i in range(10)]
    for i in range(imgs.shape[0]):
        array = (imgs[i].reshape((28, 28)) * 255).astype(np.uint8)
        cnt[labels[i]] += 1
        img = Image.fromarray(array, 'L')
        img.save(str(labels[i]) + '/' + str(cnt[labels[i]]) + '.jpg')
    ```
    ```
    从以上代码中可以看出本脚本需要安装tensorflow，使用tensorflow的read_data_sets接口来获取mnist数据集，故需安装tensorflow（如有则跳过）
    ```

    - step2:安装tensorflow(执行以下命令)

    ```bash
    pip --default-timeout=1000 install tensorflow -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
    ```
    ```
    以上安装了最新版的tensorflow，在最新版的库中是没有examples，故会报错无tensorflow.examples及tensorflow.examples.tutorials，此问题参考以下链接修复：
    https://blog.csdn.net/qq_40846862/article/details/124035370
    https://blog.csdn.net/weixin_49883619/article/details/121879790
    ```
    - step3:创建MNIST_data_folder目录用以存放所下载的数据集
    - step4:执行命令python get_dataset.py将其以图片的形式提取出来（train、test、val分别执行一次）
- 查看结果：
```
结构树形式如下：
```
```bash
    ├── MNIST_data_folder
    │   ├── t10k-images-idx3-ubyte.gz
    │   ├── t10k-labels-idx1-ubyte.gz
    │   ├── train-images-idx3-ubyte.gz
    │   └── train-labels-idx1-ubyte.gz
    ├── dataset
    │   ├── test
    │   ├── train
    │   └── val
    └── get_dataset.py
```
```
每次生成的图片都将其分别保存至命名为0-9的文件夹中，将其进行整理为train、test、val三个文件夹
```
```
最终保存的结果以表格形式列出：
```
| 数据集 |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  总计  |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|--------|
| 训练集 | 5444 | 6179 | 5470 |  5638  |  5307  |  4987  |  5417  |  5715  |  5389  |  5454  |  55000  |
| 验证集 | 479 | 563 | 488 |  493  |  535  |  434  |  501  |  550  |  462  |  495  |  5000  |
| 测试集 | 980 | 1135 | 1032 |  1010  |  982  |  892  |  958  |  1028  |  974  |  1009  |  10000  |
| 总计 | 6903 | 7877 | 6990 |  7141  |  6824  |  6313  |  6876  |  7293  |  6825  |  6958  |  70000  |

- 结果整理:
```
配置txt文件，用于训练和验证集使用
```
```
最终，数据集保存在mnist_dataset目录下，数据集txt配置文件存放至data_example目录下，训练及推理的json文件保存至configs目录下
```
### 3.2 训练前准备之配置文件的准备（config.json和infer_config.json的准备）
```
准备json文件，并将其存放至configs目录下，关于json文件中的信息如下所示：
```
- configuration

| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| model_module_name               | eg: vgg_module                                                            |
| model_net_name                  | net function name in module, eg:vgg16                                     |
| gpu_id                          | eg: single GPU: "0", multi-GPUs:"0,1,3,4,7"                                                           |
| async_loading                   | make an asynchronous copy to the GPU                                      |
| is_tensorboard                  | if use tensorboard for visualization                                      |
| evaluate_before_train           | evaluate accuracy before training                                         |
| shuffle                         | shuffle your training data                                                |
| data_aug                        | augment your training data                                                |
| img_height                      | input height                                                              |
| img_width                       | input width                                                               |
| num_channels                    | input channel                                                             |
| num_classes                     | output number of classes                                                  |
| batch_size                      | train batch size                                                          |
| dataloader_workers              | number of workers when loading data                                       |
| learning_rate                   | learning rate                                                             |
| learning_rate_decay             | learning rate decat rate                                                  |
| learning_rate_decay_epoch       | learning rate decay per n-epoch                                           |
| train_mode                      | eg:  "fromscratch","finetune","update"                                    |
| file_label_separator            | separator between data-name and label. eg:"----"                          |
| pretrained_path                 | pretrain model path                                                       |
| pretrained_file                 | pretrain model name. eg:"alexnet-owt-4df8aa71.pth"                        |
| pretrained_model_num_classes    | output number of classes when pretrain model trained. eg:1000 in imagenet |
| save_path                       | model path when saving                                                    |
| save_name                       | model name when saving                                                    |
| train_data_root_dir             | training data root dir                                                    |
| val_data_root_dir               | testing data root dir                                                     |
| train_data_file                 | a txt filename which has training data and label list                     |
| val_data_file                   | a txt filename which has testing data and label list                      |

```
比较重要的参数信息是：输入尺寸（大多数设置为64的倍数，比如128，192，256等）、预训练权重及获得训练权重的保存路径及名字、训练集和验证集的存放路径及txt路径
```
### 3.3 修改代码

- PART1:
```python
config = process_config(os.path.join(os.path.dirname(__file__), 'configs', 'mnist_config.json'))
```
修改为自己的json文件即可。
- PART2：
```
我使用了GeM_ResNet_AVP网络，所以我要修改这个网络结构对应的类
```
```python
def __init__(self, model_type = "resnet34", in_size = 128 ,num_classes=10):
```
修改上面这行，将in_size改为自己定义的模型输入尺寸，将num_classes定义为自己所划分类别即可。

### 3.4 开始训练
- 执行以下代码即可：
```bash
python train.py
```

## 4. 测试推理
### 4.1 修改代码
```
主要修改inference.py文件中的传参，包括json名字、测试图片集路径、网络名称、分类类别数、模型尺寸等
```
### 4.2 执行如下代码
```bash
python inference.py
```
```
注：inference_uncertain.py增加了不确定性因子，用于减小误差，可以参考使用。（只有部署使用，和训练无关）
```

## 5. 转模型
```
这里设置了三种转模型功能，包括json(用于TensorRT推理)、pt（jit）以及onnx（用于可视化查看学习网络结构）
其中，json部分调用函数在pytorch2json目录下，使用方法见开关设置。
将转出的json进一步按照TensorRT需求进一步修改即可配合weights小权重文件使用。（这个转出来没有add算子，是因为结构将resnet34包起来了，add需要自己对照onnx加到json中）
```
```
初步测试了，使用TensorRT推理时，batchsize为32时，耗时约2.8ms。
```

## 6. 可视化工具
- 使用tensorboard可视化loss
```
首先进入./logs目录，然后执行如下命令：
tensorboard --logdir . --bind_all
此时会在vscode终端出现如下内容：
```
```bash
(base) root@24956317f57e:/home/image_classification_pytorch/logs# tensorboard --logdir . --bind_all
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

TensorBoard 2.9.0 at http://24956317f57e:6006/ (Press CTRL+C to quit)
```
```
如上内容中：http://24956317f57e:6006/表示容器24956317f57e中的端口6006，然后在端口那栏添加端口在本地浏览器打开即可。
```

## 7. 写在最后
训练过程中所遇问题：
```python
batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
```
报错以上代码，是因为python版本太高，此版本的python中存在async保留关键字与cuda混淆掉，所以报错。改为如下即可(不加异步拷贝操作)：
```python
batch_x, batch_y = batch_x.cuda( ), batch_y.cuda( )
```
参考链接： https://qa.1r1g.com/sf/ask/3724107411/
