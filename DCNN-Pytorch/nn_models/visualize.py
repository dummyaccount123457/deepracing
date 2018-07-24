import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import string
import argparse 
import nn_models.Models as models
import matplotlib.pyplot as plt
import cv2
import data_loading.data_loaders as loaders
from tqdm import tqdm as tqdm

def get_fmaps(network,input,throttle,brake):
    print(input.shape)
    output = network(input,throttle,brake)
    print(output.size)
    cvt2pil = transforms.ToPILImage()
    plt.imshow(cvt2pil(output.squeeze().data.cpu()))
    plt.show()
    return

def load_image(filepath):
    #print(filepath)
    img = cv2.imread(filepath)
    #print(img)
    #print(type(img))
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize AdmiralNet")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--layer", type=str, required=True)
    args = parser.parse_args()
    
    annotation_dir, annotation_file = os.path.split(args.annotation_file)
    model_dir, model_file = os.path.split(args.model_file)
    config_path = os.path.join(model_dir,'config.pkl')
    config_file = open(config_path,'rb')
    config = pickle.load(config_file)
    #print(config)
    
    prefix, _ = annotation_file.split(".")
    prefix = prefix + config['file_prefix']

    gpu = int(config['gpu'])
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    size = (66,200)
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])
    optical_flow = bool(config.get('optical_flow',''))
    label_scale = float(config['label_scale'])
    
    cell_type = 'lstm'
    network = models.AdmiralNet(cell=cell_type,context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, optical_flow=optical_flow)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(pytorch_total_params)

    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        dataset = loaders.F1SequenceDataset(annotation_dir,annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, use_float32=True, label_transformation = label_transformation, optical_flow=optical_flow)
    else:
        network.double()
        dataset = loaders.F1SequenceDataset(annotation_dir, annotation_file,(66,200),\
        context_length=context_length, sequence_length=sequence_length, label_transformation = label_transformation, optical_flow=optical_flow)
    
    if(gpu>=0):
        network = network.cuda(gpu)
    if optical_flow:
        if((not os.path.isfile("./" + prefix+"_opticalflows.pkl")) or (not os.path.isfile("./" + prefix+"_opticalflowannotations.pkl"))):
            dataset.read_files_flow()
            dataset.write_pickles(prefix+"_opticalflows.pkl",prefix+"_opticalflowannotations.pkl")
        else:  
            dataset.read_pickles(prefix+"_opticalflows.pkl",prefix+"_opticalflowannotations.pkl")
    else:
        if((not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
            dataset.read_files()
            dataset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
        else:  
            dataset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")

    dataset.img_transformation = config['image_transformation']
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 4)
    for (idx, (inputs, throttle, brake,_, labels)) in tqdm(enumerate(loader)):
        if(idx!=4):
            continue
        else:
            if(gpu>=0):
                inputs = inputs.cuda(gpu)
                throttle = throttle.cuda(gpu)
                brake= brake.cuda(gpu)
                labels = labels.cuda(gpu)
            network.eval()
            f = torch.nn.Sequential(*list(network.children())[0][:6])
            

            get_fmaps(network,inputs,throttle,brake)
            break

if __name__ == '__main__':
    main()




