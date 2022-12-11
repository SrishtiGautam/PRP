# PRP

The official PyTorch implementation of "This looks more like that: Enhancing Self-Explaining Models by Prototypical Relevance Propagation" [Pattern Recognition 2022](https://www.sciencedirect.com/science/article/pii/S0031320322006513) by Srishti Gautam,  Marina MC Höhne, Stine Hansen Robert Jenssen, Michael Kampffmeyer.

The code is built upon ProtoPNet's official implementation (https://github.com/cfchen-duke/ProtoPNet) and LRP implementation from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet.

### Requirements
- matplotlib 
- numpy
- pytorch
- scikit-image
- pillow 


### Instructions for generating a PRP map for a test image
1. Train a ProtoPNet model with your choice of dataset and save the model.
2. Edit the settings file for changing the required parameters such as trained model's path and name, test image's path and name, prototype number for the PRP map, and base architecture used while training ProtoPNet.
3. Run main.py for generating the PRP map.
