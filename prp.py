#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:43:52 2020

@author: srishtigautam

Code built upon LRP code from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet
"""

from __future__ import print_function, division

from torchvision import datasets, models, transforms

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from resnet_features import BasicBlock, Bottleneck, ResNet_features
import torch.nn.functional as F

#############
# from getimagenetclasses import *
# from dataset_imagenet2500 import dataset_imagenetvalpart_nolabels
from heatmaphelpers import *

from lrp_general6 import *
from collections import OrderedDict
import math

import os

layers = []


##########
##########
##########
##########
##########
##########


# partial replacement of BN, use own classes, no pretrained loading


class Cannotloadmodelweightserror(Exception):
    pass


class Modulenotfounderror(Exception):
    pass


class BasicBlock_fused(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_fused, self).__init__(inplanes, planes, stride, downsample)

        # own
        self.elt = sum_stacked2()  # eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        out = self.elt(torch.stack([out, identity], dim=0))  # self.elt(out,identity)
        out = self.relu(out)

        return out


class Bottleneck_fused(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_fused, self).__init__(inplanes, planes, stride, downsample)

        # own
        self.elt = sum_stacked2()  # eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.elt(torch.stack([out, identity], dim=0))  # self.elt(out,identity)
        out = self.relu(out)

        return out


class ResNet_canonized(ResNet_features):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_canonized, self).__init__(block, layers, num_classes=1000, zero_init_residual=False)

        ######################
        # change
        ######################
        # own
        # self.avgpool = nn.AvgPool2d(kernel_size=7,stride=7 ) #nn.AdaptiveAvgPool2d((1, 1))

    # runs in your current module to find the object layer3.1.conv2, and replaces it by the obkect stored in value (see         success=iteratset(self,components,value) as initializer, can be modified to run in another class when replacing that self)
    def setbyname(self, name, value):

        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                # print('found!!', components[0])
                # exit()
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split('.')
        success = iteratset(self, components, value)
        return success

    def copyfromresnet(self, net, lrp_params, lrp_layer2method):
        # assert( isinstance(net,ResNet))

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            print('at src_module_name', src_module_name)

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                print('is Linear')
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                print(wrapped)
                # exit()
                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                foundsth = True
                print('is Conv2d')
                last_src_module_name = src_module_name
                last_src_module = src_module
            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # conv-bn chain
                foundsth = True
                print('is BatchNorm2d')

                if (True == lrp_params['use_zbeta']) and (last_src_module_name == 'conv1'):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)
                print(wrapped)
                # exit()

                if False == self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy")

                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                print(wrapped)
                # exit()
                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            # if False== foundsth:
            #  print('!untreated layer')
            print('\n')

        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in self.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                print(wrapped)
                # exit()
                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                print(wrapped)
                # exit()
                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)

        for target_module_name, target_module in self.named_modules():
            if target_module_name not in updated_layers_names:
                print('not updated:', target_module_name)


class addon_canonized(nn.Module):

    def __init__(self):
        super(addon_canonized, self).__init__()
        self.addon = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.Sigmoid()
        )


def _addon_canonized(pretrained=False, progress=True, **kwargs):
    model = addon_canonized()
    # if pretrained:
    #     raise Cannotloadmodelweightserror("explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model")
    return model


def _resnet_canonized(arch, block, layers, **kwargs):
    model = ResNet_canonized(block, layers, **kwargs)
    # if pretrained:
    #     raise Cannotloadmodelweightserror("explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model")
    return model


def resnet18_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet18', BasicBlock_fused, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet50', Bottleneck_fused, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet34_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet34', BasicBlock_fused, [3, 4, 6, 3], **kwargs)


def resnet152_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet152', Bottleneck_fused, [3, 8, 36, 3], **kwargs)


def resnet101_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet101', Bottleneck_fused, [3, 4, 23, 3], **kwargs)



class sum_lrp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # *values unpacks the list

        print('ctx.needs_input_grad', ctx.needs_input_grad)
        # exit()
        print('sum custom forward')
        return torch.sum(x, dim=(1, 2, 3))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        # print('len(grad_output)',len(grad_output),grad_output[0].shape)

        input_ = ctx.saved_tensors
        X = input_.clone().detach().requires_grad_(True)
        # R= lrp_backward(_input= X , layer = layerclass , relevance_output = grad_output[0], eps0 = 1e-12, eps=0)
        with torch.enable_grad():
            Z = torch.sum(X, dim=(1, 2, 3))
        relevance_output_data = grad_output[0].clone().detach().unsqueeze(0)
        # Z.backward(relevance_output_data)
        # R = X.grad
        R = relevance_output_data * X / Z
        # print('sum R', R.shape)
        # exit()
        return R, None


def generate_prp_all(dataloader, model, foldername, device):
    model.train(False)

    for pno in range(model.num_prototypes):
        i = 0
        for data in dataloader:
            # get the inputs
            inputs = data[0]
            fns = data[2]

            inputs = inputs.to(device).clone()

            inputs.requires_grad = True

            with torch.enable_grad():

                conv_features = model.conv_features(inputs)

                newl2 = l2_lrp_class.apply
                similarities = newl2(conv_features, model)

                # global max pooling
                min_distances = model.max_layer(similarities)

                min_distances = min_distances.view(-1, model.num_prototypes)

            '''For individual prototype'''
            (min_distances[:, pno]).backward()

            rel = inputs.grad.data
            print(fns)
            print("\n")
            #
            imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'), folder=foldername+str(pno)+"/", name=fns[0].split("/")[4])


def generate_prp_image(inputs, pno, model, device):
    model.train(False)
    inputs = inputs.to(device).clone()

    inputs.requires_grad = True

    with torch.enable_grad():
        conv_features = model.conv_features(inputs)

        newl2 = l2_lrp_class.apply
        similarities = newl2(conv_features, model)

        # global max pooling
        min_distances = model.max_layer(similarities)

        min_distances = min_distances.view(-1, model.num_prototypes)

    '''For individual prototype'''
    (min_distances[:, pno]).backward()

    rel = inputs.grad.data
    print("\n")
    #
    prp = imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'))

    return prp




class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def heatmap(R, sx, sy, name):
    # b = 10*np.abs(R).mean()
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    plt.savefig(name)
    plt.clf()



def setbyname(obj, name, value):

    def iteratset(obj, components, value):

        if not hasattr(obj, components[0]):
            print(components[0])
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(obj, components, value)
    return success




base_architecture_to_features = {'resnet18': resnet18_canonized,
                                 'resnet34': resnet34_canonized,
                                 'resnet50': resnet50_canonized,
                                 'resnet101': resnet101_canonized,
                                 'resnet152':resnet152_canonized}



def PRPCanonizedModel(ppnet,base_arch):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = base_architecture_to_features[base_arch](pretrained=False)
    model = model.to(device)


    lrp_params_def1 = {
        'conv2d_ignorebias': True,
        'eltwise_eps': 1e-6,
        'linear_eps': 1e-6,
        'pooling_eps': 1e-6,
        'use_zbeta': True,
    }

    lrp_layer2method = {
        'nn.ReLU': relu_wrapper_fct,
        'nn.Sigmoid': sigmoid_wrapper_fct,
        'nn.BatchNorm2d': relu_wrapper_fct,
        'nn.Conv2d': conv2d_beta0_wrapper_fct,
        'nn.Linear': linearlayer_eps_wrapper_fct,
        'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
        'nn.MaxPool2d': maxpool2d_wrapper_fct,
        'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct
    }

    model.copyfromresnet(ppnet.features, lrp_params=lrp_params_def1, lrp_layer2method=lrp_layer2method)
    model = model.to(device)
    ppnet.features = model

    add_on_layers = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
        nn.Sigmoid()
    )

    conv_layer1 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(device)
    conv_layer1.weight.data = ppnet.ones

    wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer1), lrp_params_def1, lrp_layer2method)
    conv_layer1 = wrapped

    conv_layer2 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(device)
    conv_layer2.weight.data = ppnet.prototype_vectors

    wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer2), lrp_params_def1, lrp_layer2method)
    conv_layer2 = wrapped

    relu_layer = nn.ReLU().to(device)
    wrapped = get_lrpwrapperformodule(copy.deepcopy(relu_layer), lrp_params_def1, lrp_layer2method)
    relu_layer = wrapped

    wrapped = get_lrpwrapperformodule(copy.deepcopy(ppnet.last_layer), lrp_params_def1, lrp_layer2method)
    last_layer = wrapped


    add_on_layers = _addon_canonized()
    for src_module_name, src_module in ppnet.add_on_layers.named_modules():
        if isinstance(src_module, nn.Conv2d):
            print(hasattr(add_on_layers.addon, src_module_name))
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

        if isinstance(src_module, nn.ReLU):
            print(hasattr(add_on_layers.addon, src_module_name))
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

        if isinstance(src_module, nn.Sigmoid):
            print(hasattr(add_on_layers.addon, src_module_name))
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

    ppnet.max_layer = torch.nn.MaxPool2d((7, 7), return_indices=False)

    ## Maxpool
    ppnet.max_layer = get_lrpwrapperformodule(copy.deepcopy(ppnet.max_layer), lrp_params_def1, lrp_layer2method)

    add_on_layers = add_on_layers.to(device)
    ppnet.add_on_layers = add_on_layers.addon

    ppnet.conv_layer1 = conv_layer1
    ppnet.conv_layer2 = conv_layer2
    ppnet.relu_layer = relu_layer
    ppnet.last_layer = last_layer

    return ppnet





