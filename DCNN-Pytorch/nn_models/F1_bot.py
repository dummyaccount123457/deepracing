import numpy as np
import py_vjoy
import argparse
from PIL import ImageGrab
import cv2
import nn_models.Models as models
import torch
import torch.nn as nn 
import pickle
import os
import string
import time
import pyf1_datalogger
import torchvision.transforms as transforms
import imutils.annotation_utils
import numpy_ringbuffer
import data_loading.data_loaders as dataloaders
import PIL
import PIL.Image as pilImg
def grab_screen(dl):
    return dl.read()
def fill_buffer(buffer, dataloader, context_length, dt=0.0, interp = cv2.INTER_AREA, totens = transforms.ToTensor(), normalize=None ):
    pscreen = grab_screen(dataloader)
    pscreen_grey = cv2.cvtColor(pscreen,cv2.COLOR_BGRA2GRAY)
    pscreen_grey = cv2.resize(pscreen_grey,(200,66), interpolation=interp)
    while(buffer.shape[0]<context_length):
        cv2.waitKey(dt)
        screen = grab_screen(dataloader)
        screen_resize = cv2.resize( screen, ( 200, 66 ) , interpolation=interp)
        screen_grey = cv2.cvtColor(screen_resize,cv2.COLOR_BGRA2GRAY)
        flow = cv2.calcOpticalFlowFarneback(pscreen_grey,screen_grey, None, 0.5, 3, 20, 8, 5, 1.2, 0)
        flow_ = flow.transpose(2, 0, 1).astype(np.float32)
        screen_ = totens(screen_resize[:,:,0:3]).float().numpy()
        print(flow_.shape)
        print(screen_.shape)
        im = np.concatenate((screen_,flow_),axis=0)
        if normalize is not None:
            im = normalize( torch.from_numpy(im) ).numpy()
        buffer.append(im)
        pscreen_grey = screen_grey
    return screen_grey

    
def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("--model_file", type=str, required=True)
    args = parser.parse_args()
    
    model_dir, model_file = os.path.split(args.model_file)
    config_path = os.path.join(model_dir,'config.pkl')
    config_file = open(config_path,'rb')
    config = pickle.load(config_file)
    print(config)
    model_prefix, _ = model_file.split(".")

    gpu = int(config['gpu'])
    use_float32 = bool(config['use_float32'])
    label_scale = float(config['label_scale'])
    context_length = int(config['context_length'])
    sequence_length = int(config['sequence_length'])
    hidden_dim = int(config['hidden_dim'])
    optical_flow = bool(config.get('optical_flow',''))
    image_transformation = config.get('image_transformation',None)
    rnn_cell_type='lstm'
    network = models.AdmiralNet(cell=rnn_cell_type,context_length = context_length, sequence_length=sequence_length, hidden_dim = hidden_dim, use_float32 = use_float32, gpu = gpu, input_channels=5)
    state_dict = torch.load(args.model_file)
    network.load_state_dict(state_dict)
    network=network.float()
    network=network.cuda(0)
    print(network)
    vjoy_max = 32000
    
    throttle = torch.Tensor(1,10)
    brake = torch.Tensor(1,10)
    if(use_float32):
        network.float()
    else:
        network.double()
    if(gpu>=0):
        network = network.cuda(gpu)
    network.eval()
    vj = py_vjoy.vJoy()
    vj.capture(1) #1 is the device ID
    vj.reset()
    js = py_vjoy.Joystick()
    js.setAxisXRot(int(round(vjoy_max/2))) 
    js.setAxisYRot(int(round(vjoy_max/2))) 
    vj.update(js)
    time.sleep(2)
    inputs = []
    '''
    '''
    wheel_pred = pilImg.open('steering_wheel.png')
    wheelrows_pred = 66
    wheelcols_pred = 66
    resize_wheel = transforms.Resize((wheelcols_pred,wheelrows_pred))
    wheel_pred = resize_wheel(wheel_pred)
    toNP_alpha = transforms.Compose([transforms.Lambda(lambda img : np.array(img) ) , transforms.Lambda(lambda img : cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA ) )  ] )
    buffer = numpy_ringbuffer.RingBuffer(capacity=context_length, dtype=(np.float32, (5,66,200) ) )

    dt = 12
    debug=True
    app="F1 2017"
    dl = pyf1_datalogger.ScreenVideoCapture()
    dl.open(app,0,200,1700,300)
    interp = cv2.INTER_AREA
    if debug:
        cv2.namedWindow(app, cv2.WINDOW_AUTOSIZE)
    totens = transforms.ToTensor()
    resize = transforms.Resize((66,200))
    toNP = transforms.Compose([transforms.Lambda(lambda img : np.array(img) ) , transforms.Lambda(lambda img : cv2.cvtColor(img, cv2.COLOR_RGB2BGR ) )  ] )
    toPIL = transforms.Compose([transforms.Lambda(lambda img : cv2.cvtColor(img, cv2.COLOR_BGR2RGB ) ) , transforms.ToPILImage() ] )
    toPIL_alpha = transforms.Compose([transforms.Lambda(lambda img : cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA ) ) , transforms.ToPILImage() ] )
    print(context_length)
    pscreen_grey = fill_buffer(buffer,dl,context_length,dt=dt,interp=interp, normalize = image_transformation)
    buffer_torch = torch.zeros((1,context_length,5,66,200), dtype=torch.float32)
    while(True):
        cv2.waitKey(dt)
        screen = grab_screen(dl)
        screen_pil = toPIL_alpha(screen)
        screen_resize = resize(screen_pil)
        screen_resizenp = toNP_alpha(screen_resize)
        screen_grey = cv2.cvtColor(screen_resizenp, cv2.COLOR_BGRA2GRAY)
        flow = cv2.calcOpticalFlowFarneback(pscreen_grey,screen_grey, None, 0.5, 3, 20, 8, 5, 1.2, 0)
        screen_ = totens(screen_resizenp[:,:,0:3]).float()
        flow_ = torch.from_numpy(flow.transpose(2, 0, 1).astype(np.float32))
        im = torch.cat( ( screen_, flow_ ) ,dim=0)
        if image_transformation is not None:
            im = image_transformation( im )
        buffer.append( im.numpy() )
        pscreen_grey = screen_grey
        buffer_arr = np.array(buffer)
        buffer_torch[0] = torch.from_numpy(buffer_arr)
        buffer_torch=buffer_torch.cuda(gpu)
        #print("Input Size: " + str(buffer_torch.size()))
        outputs = network(buffer_torch, throttle=None, brake=None )
        angle = outputs[0][0].item() +0.04
        print("Output: " + str(angle))
        scaled_pred_angle = 180.0*angle
        wheel_pred_rotated = transforms.functional.rotate(wheel_pred,scaled_pred_angle)
        wheel_np = toNP_alpha(wheel_pred_rotated)
        background = screen
        out_size = background.shape
        overlayed_pred = imutils.annotation_utils.overlay_image(background,wheel_np)
        if debug:
            cv2.imshow(app,overlayed_pred)
        '''
        vjoy_angle = -angle*vjoy_max + vjoy_max/2.0
        js.setAxisXRot(int(round(vjoy_angle))) 
        js.setAxisYRot(int(round(vjoy_angle))) 
        vj.update(js)
        '''
        
    print(buffer.shape)             
        
if __name__ == '__main__':
    main()