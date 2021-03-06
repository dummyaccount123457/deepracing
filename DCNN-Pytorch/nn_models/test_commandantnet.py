import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import data_loading.data_loaders as loaders
import numpy.random
import torch, random
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm as tqdm
import pickle
from datetime import datetime
import os
import string
import argparse
from random import randint
from datetime import datetime
import imutils.annotation_utils
from data_loading.image_loading import load_image
import torchvision.transforms as transforms

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    plot = args.plot
    annotation_dir, annotation_file = os.path.split(args.annotation_file)
    model_dir, model_file = os.path.split(args.model_file)
    config_path = os.path.join(model_dir,'config.pkl')
    config_file = open(config_path,'rb')
    config = pickle.load(config_file)
    print(config)
    model_prefix, _ = model_file.split(".")
   # return

    gpu = int(config['gpu'])
    optical_flow = bool(config.get('optical_flow',''))
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    prefix, _ = annotation_file.split(".")
    prefix = prefix + config['file_prefix'] + 'commandant'
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])
    size = (125, 400)
    network = models.CommandantNet(context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, optical_flow=optical_flow)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    print(network)
    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        trainset = loaders.F1SequenceDataset(annotation_dir,annotation_file,size,context_length=context_length, sequence_length=sequence_length, use_float32=True, label_transformation = label_transformation, optical_flow=optical_flow)
    else:
        network.double()
        trainset = loaders.F1SequenceDataset(annotation_dir, annotation_file,size,context_length=context_length, sequence_length=sequence_length, label_transformation = label_transformation, optical_flow=optical_flow)
    
    if(gpu>=0):
        network = network.cuda(gpu)
    if optical_flow:
        if((not os.path.isfile("./" + prefix+"_commandantopticalflows.pkl")) or (not os.path.isfile("./" + prefix+"_commandantopticalflowannotations.pkl"))):
            trainset.read_files_flow()
            trainset.write_pickles(prefix+"_commandantopticalflows.pkl",prefix+"_commandantopticalflowannotations.pkl")
        else:  
            trainset.read_pickles(prefix+"_commandantopticalflows.pkl",prefix+"_commandantopticalflowannotations.pkl")
    else:
        if((not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
            trainset.read_files()
            trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
        else:  
            trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    ''' '''

    mean,stdev = trainset.statistics()
    mean_ = torch.from_numpy(mean)
    stdev_ = torch.from_numpy(stdev)
    if use_float32:
        mean_.float()
        stdev_.float()
    trainset.img_transformation = config['image_transformation']
    if plot:
        batch_size = 1
    else:
        batch_size = 32
    loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = False, num_workers = 0)
    cum_diff = 0.0
    t = tqdm(enumerate(loader))
    network.eval()
    predictions=[]
    ground_truths=[]
    losses=[]
    criterion = nn.MSELoss()
    if(gpu>=0):
        criterion = criterion.cuda(gpu)
    if args.write_images:
        imdir = "admiralnet_prediction_images_" + model_prefix
        os.mkdir(imdir)
        annotation_file = open(args.annotation_file,'r')
        annotations = annotation_file.readlines()
        annotation_file.close()
        im,_,_,_,_ = annotations[0].split(",")
        background = cv2.imread(os.path.join(annotation_dir,'raw_images',im),cv2.IMREAD_UNCHANGED)
        out_size = background.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoout = cv2.VideoWriter(os.path.join(imdir,"video.avi") ,fourcc, 60.0, (out_size[1], out_size[0]),True)
        wheel = cv2.imread('steering_wheel.png',cv2.IMREAD_UNCHANGED)
        wheelrows = 150
        wheelcols = 150
        wheel = cv2.resize(wheel, (wheelcols,wheelrows), interpolation = cv2.INTER_CUBIC)
    for idx,(inputs, previous_control, labels) in t:
        if(gpu>=0):
            previous_control = previous_control.cuda(gpu)
            inputs = inputs.cuda(gpu)
            labels = labels.cuda(gpu)
        pred = torch.div(network(inputs,previous_control),label_scale)
        if plot:
            if pred.shape[1] == 1:
                angle = pred.item()
                ground_truth = labels.item()
            else:
                angle = pred.squeeze()[0].item()
                ground_truth = labels.squeeze()[0].item()
            predictions.append(angle)
            ground_truths.append(ground_truth)
            t.set_postfix(angle = angle, ground_truth = ground_truth)
        loss = criterion(pred, labels)
        losses.append(torch.mean(loss).item())
       # print("Ground Truth: %f. Prediction: %f.\n" %(scaled_ground_truth, scaled_angle))
        if args.write_images:
            scaled_angle = 180.0*angle
            M = cv2.getRotationMatrix2D((wheelrows/2,wheelcols/2),scaled_angle,1)
            wheel_rotated = cv2.warpAffine(wheel,M,(wheelrows,wheelcols))
            numpy_im = np.transpose(trainset.images[idx],(1,2,0)).astype(np.float32)
           # print(numpy_im.shape)
            im,_,_,_,_ = annotations[idx].split(",")
            background = cv2.imread(os.path.join(annotation_dir,'raw_images',im),cv2.IMREAD_UNCHANGED)
            out_size = background.shape
            #print(background.shape)
            overlayed = imutils.annotation_utils.overlay_image(background,wheel_rotated,int((out_size[1]-wheelcols)/2),int((out_size[0]-wheelcols)/2))
            name = "ouput_image_" + str(idx) + ".png"
            output_path = os.path.join(imdir,name)
            cv2.imwrite(output_path,overlayed)
            videoout.write(overlayed)
        '''
        '''
    predictions_array = np.array(predictions)
    ground_truths_array = np.array(ground_truths)
    diffs = np.subtract(predictions_array,ground_truths_array)
    rms = np.sqrt(np.mean(np.array(losses)))
    print("RMS Error: ", rms)
    if args.plot:
        from scipy import stats
        import matplotlib.pyplot as plt
        t = np.linspace(0,len(loader)-1,len(loader))
        plt.plot(t,predictions_array,'r')
        plt.plot(t,ground_truths_array,'b')
        #plt.plot(t,diffs)
        plt.show()
if __name__ == '__main__':
    main()
