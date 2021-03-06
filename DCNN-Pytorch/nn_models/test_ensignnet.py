import cv2
import numpy as np
import nn_models
import data_loading.image_loading as il
import nn_models.Models as models
import data_loading.data_loaders_old as loaders
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
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    prefix, _ = annotation_file.split(".")
    prefix = prefix + config['file_prefix']
    size = (66, 200)
    criterion = nn.MSELoss()
    network = models.EnsignNet()
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    print(network)
    if(label_scale == 1.0):
        label_transformation = None
    else:
        label_transformation = transforms.Compose([transforms.Lambda(lambda inputs: inputs.mul(label_scale))])
    if(use_float32):
        network.float()
        trainset = loaders.F1Dataset(annotation_dir,annotation_file, size, use_float32=True, label_transformation = label_transformation)
    else:
        network.double()
        trainset = loaders.F1Dataset(annotation_dir, annotation_file, size, label_transformation = label_transformation)
    
    if(gpu>=0):
        criterion = criterion.cuda(gpu)
        network = network.cuda(gpu)
    if((not os.path.isfile("./" + prefix+"_images.pkl")) or (not os.path.isfile("./" + prefix+"_annotations.pkl"))):
        trainset.read_files()
        trainset.write_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    else:  
        trainset.read_pickles(prefix+"_images.pkl",prefix+"_annotations.pkl")
    ''' '''

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
    for idx,(inputs, throttle, brake, labels) in t:
        if(gpu>=0):
            inputs = inputs.cuda(gpu)
            labels = labels.cuda(gpu)
        pred = torch.div(network(inputs),label_scale)
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
    log_name = "ouput_log.txt"
    log_output_path = os.path.join(imdir,log_name)
    log = list(zip(ground_truths_array,predictions_array))
    with open(log_output_path, "a") as myfile:
        for x in log:
            log_item = [x[0],x[1]]
            myfile.write("{0},{1}\n".format(log_item[0],log_item[1]))
    diffs = np.subtract(predictions_array,ground_truths_array)
    rms = np.sqrt(np.mean(np.array(losses)))
    nrms = np.sqrt(np.mean(np.divide(np.square(np.array(losses)),np.multiply(np.mean(np.array(predictions)),np.mean(np.array(ground_truths))))))
    print("RMS Error: ", rms)
    print("RMS Error: ", nrms)
    
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
