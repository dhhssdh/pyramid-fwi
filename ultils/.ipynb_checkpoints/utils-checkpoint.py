# Common utility functions.
# Portions of this code are adapted from the implementation of
# Fangshuyang (yangfs@hit.edu.cn).
import torch
import deepwave
import numpy as np
import scipy
import scipy.io as spio
from torch import autograd
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import math
from math import exp
from IPython.core.debugger import set_trace
import scipy.stats
import warnings
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)

from scipy.signal import sosfilt
from scipy.signal import zpk2sos
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
#from ParamConfig import *


def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency.
    
    Args:
        freq: A float specifying the central frequency of the wavelet
        length: An integer specifying the number of time steps to use
        dt: A float specifying the time interval between time steps
        peak_time: A float specifying the time (in time units) at which the
                   peak amplitude of the wavelet occurs

    Returns:
        A 1D Numpy array of length 'length' containing a Ricker wavelet
    """
    t = (np.arange(length) * dt - peak_time).astype(np.float32)
    y = (1 - 2 * np.pi**2 * freq**2 * t**2) \
            * np.exp(-np.pi**2 * freq**2 * t**2)
    return y


def createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims,source_spacing, receiver_spacing,source_depth,receiver_depth):
    """
        Create arrays containing the source and receiver locations
        Args:
            num_shots: nunmber of shots
            num_sources_per_shot: number of sources per shot
            num_receivers_per_shot： number of receivers per shot
            num_dims: dimension of velocity model
        return:
            x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
            x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions] 
    """    
    x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    if source_depth != 0:
        x_s[:, 0, 0] = source_depth        
    x_s[:, 0, 1] = torch.arange(1,num_shots+1).float() * source_spacing
    x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    if receiver_depth != 0:
        x_r[:, :, 0] = receiver_depth
    x_r[0, :, 1] = torch.arange(1,num_receivers_per_shot+1).float() * receiver_spacing
    x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

    return x_s, x_r


def createSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots, num_sources_per_shot):
    """
        Create true source amplitudes [nt, num_shots, num_sources_per_shot]
        This is implemented by numpy
        Args:
            peak_freq : frequency for source
            peak_source_time: delay

        return:
            source_amplitudes

    """
    source_amplitudes_true = np.tile(ricker(peak_freq, nt, dt, peak_source_time).reshape(-1, 1, 1),[1,num_shots, num_sources_per_shot])
    
    
    return source_amplitudes_true



def createInitialModel(model_true, gfsigma, lipar, fix_value_depth, device):
    """
        Create 2D initial guess model for inversion ('line','lineminmax','const','GS')
    """
    assert gfsigma in ['line','lineminmax','constant','GS']
    model_true = model_true.cpu().detach().numpy()   
    shape = model_true.shape
    if fix_value_depth > 0:
        const_value = model_true[:fix_value_depth,:]
   
    if gfsigma == 'line':
    # generate the line increased initial model
        value = np.linspace(model_true[fix_value_depth,np.int64(shape[1]/2)], \
                            model_true[-1,np.int64(shape[1]/2)]*lipar,num=shape[0]-fix_value_depth, \
                            endpoint=True,dtype=float).reshape(-1,1)
        value = value.repeat(shape[1],axis=1)        
    elif gfsigma == 'lineminmax':
    # generate the line increased initial model (different min/max value)
        value = np.linspace(model_true.min()*lipar, \
                            model_true.max(),num=shape[0]-fix_value_depth,
                            endpoint=True,dtype=float).reshape(-1,1)
        
        value = value.repeat(shape[1],axis=1)      
    elif gfsigma == 'constant':
    # generate the constant initial model
        value = model_true[fix_value_depth, int(np.floor(shape[1] / 2))] * np.ones(shape[0]-fix_value_depth,shape[1])
    # generate the initial model by using Gaussian smoothed function
    else:
        value = scipy.ndimage.gaussian_filter(model_true[fix_value_depth:,:], sigma=gfsigma)
        
    if fix_value_depth > 0:
        model_init = np.concatenate([const_value,value],axis=0)
    else:
        model_init = value
        
    model_init = torch.tensor(model_init)
    # Make a copy so at the end we can see how far we came from the initial model
    model = model_init.clone()
   
    model = model.to(device)
    # set the requires_grad to True to update the model
    model.requires_grad = True
    print('model size:',model.size())
    
    return model


def createdata(model,dx,source_amplitudes,x_s,x_r,dt, \
               pml_width,order,survey_pad,device):
    """
        Create data depends on the velocity model 
    """
    #survey_pad = None
    prop = deepwave.scalar.Propagator({'vp': model.to(device)},dx,pml_width, \
                                      order,survey_pad)
    # the shape of receiver_amplitudes is [nt, num_shots, num_receivers_per_shot]
    receiver_amplitudes = prop(source_amplitudes.to(device), \
                               x_s.to(device), \
                               x_r.to(device),dt).cpu()

    return receiver_amplitudes
   

def createFilterSourceAmp(peak_freq,nt,dt,peak_source_time,num_shots, \
                          num_sources_per_shot,use_filter,filter_type, \
                          freqmin,freqmax,corners,df):
    """
        Create source amplitudes with filter function
        Args:
            peak_freq : frequency for source
            peak_source_time: delay
            filt_type: type of filter ('highpass','lowpass','bandpass')

        return:
            source_amplitudes_filt

    """
    
    source_amplitudes = ricker(peak_freq, nt, dt, peak_source_time)
    if use_filter:
        filt_data = seismic_filter(data=source_amplitudes, \
                           filter_type=filter_type,freqmin=freqmin, \
                           freqmax=freqmax,df=df,corners=corners)
        filt_data = filt_data
    else:
        filt_data = source_amplitudes
        
    source_amplitudes_filt = np.tile(filt_data.reshape(-1,1,1),[1, num_shots, num_sources_per_shot])
    return source_amplitudes_filt


def seismic_filter(data,filter_type,freqmin,freqmax,df,corners,zerophase=False,axis=-1):
    """
    create the fileter for removing the frequency component of seismic data 
    """
    assert filter_type.lower() in ['bandpass', 'lowpass', 'highpass']

    if filter_type == 'bandpass':
        if freqmin and freqmax and df:
            filt_data = bandpass(data, freqmin, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'lowpass':
        if freqmax and df:
            filt_data = lowpass(data, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'highpass':
        if freqmin and df:
            filt_data = highpass(data, freqmin, df, corners, zerophase, axis)
        else:
            raise ValueError
    return filt_data



    
def bandpass(data, freqmin, freqmax, df, corners, zerophase, axis):
    """
    Butterworth-Bandpass Filter.
    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)

    
def lowpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Lowpass Filter.
    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)


def highpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Highpass Filter.
    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)



def loadtruemodel(data_dir, num_dims, vmodel_dim):
    """
        Load the true model
    """
    
    if num_dims != len(vmodel_dim.reshape(-1)):
        raise Exception('Please check the size of model_true!!')
    # prefer the depth direction first, that is the shape is `[nz, (ny, (nx))]`
    if num_dims == 2:       
        model_true = (np.fromfile(data_dir, np.float32).reshape(vmodel_dim[1],vmodel_dim[0]))
        model_true = np.transpose(model_true,(1,0)) # I prefer having depth direction first
    else:
        raise Exception('Please check the size of model_true!!')
   
    model_true = torch.Tensor(model_true) # Convert to a PyTorch Tensor
    
    return model_true

def loadrcv(rcvfile,device):
    """
        Load the receiver amplitude
    """
    data_mat     = spio.loadmat(rcvfile)    
   
    receiver_amplitudes_true  = torch.from_numpy(np.float32(data_mat[str('true')]))
    
    receiver_amplitudes_true = receiver_amplitudes_true.to(device)
    return receiver_amplitudes_true 




def loadinitmodel(initfile,device):
    """
        Load initial model guess
    """
    model_mat = spio.loadmat(initfile)
    model_init = torch.from_numpy(np.float32(model_mat[str('initmodel')]))
    model =  model_init.clone().to(device)
    model.requires_grad = True 
    
    return model, model_init

def fix_model_grad(fix_value_depth,model):
    assert fix_value_depth>0
    device = model.device
    # Create Gradient mask
    gradient_mask = torch.zeros(model.shape).to(device)
    # set the [receiver_depth:,:] = 1; [:receiver_depth,:] = 0;
    gradient_mask[fix_value_depth:,:] = 1.0
    # only update the [receiver_depth:,:]
    model.register_hook(lambda grad: grad.mul_(gradient_mask))

def loadinitsource(initsafile,device):
    """
        Load initial source amplitude guess
    """
    source_mat = spio.loadmat(initsafile)
    source_init = torch.from_numpy(np.float32(source_mat[str('initsource')])).to(device)
    source_true =  torch.from_numpy(np.float32(source_mat[str('truesource')])).to(device)
       
    return source_init, source_true


    
def createlearnSNR(init_snr_guess,device):
    """
        create learned snr when amplitude is noisy and try to learn the noise
    """
    learn_snr_init = torch.tensor(init_snr_guess)
    learn_snr = learn_snr_init.clone()
    learn_snr = learn_snr.to(device)
    #set_trace()
    learn_snr.requires_grad = True
    
    return learn_snr, learn_snr_init
      

    
    
def gaussian(window_size, sigma):
    """
    gaussian filter
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    create the window for computing the SSIM
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def ComputeSSIM(img1, img2, window_size=11, size_average=True):
    """
    compute the SSIM between img1 and img2
    """
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    
    if len(img1.size()) == 2:
        d = img1.size()
        img1 = img1.view(1,1,d[0],d[1])
        img2 = img2.view(1,1,d[0],d[1])
    elif len(img1.size()) == 3:
        d = img1.size()
        img1 = img1.view(d[2],1,d[0],d[1])
        img2 = img2.view(d[2],1,d[0],d[1]) 
    else:
        raise Exception('The shape of image is wrong!!!')
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ComputeSNR(rec,target):
    """
       Calculate the SNR between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    snr = 0.0
    if len(rec.shape) == 3:
        for i in range(rec.shape[0]):
            rec_ind     = rec[i,:,:].reshape(np.size(rec[i,:,:]))
            target_ind  = target[i,:,:].reshape(np.size(rec_ind))
            s      = 10*np.log10(sum(target_ind**2)/sum((rec_ind-target_ind)**2))
            snr    = snr + s
        snr = snr/rec.shape[0]
    elif len(rec.shape) == 2:
        rec       = rec.reshape(np.size(rec))
        target    = target.reshape(np.size(rec))
        snr       = 10*np.log10(sum(target**2)/sum((rec-target)**2))
    else:
        raise Exception('Please reshape the Rec to correct Dimension!!')
    return snr

def ComputeRSNR(rec,target):
    """
       Calculate the regressed-SNR(RSNR) between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    rec_ind     = rec.reshape(np.size(rec))
    target_ind  = target.reshape(np.size(rec))
    slope,intercept, _, _, _ = scipy.stats.linregress(rec_ind,target_ind)
    r           = slope*rec_ind + intercept
    rsnr        = 10*np.log10(sum(target_ind**2)/sum((r-target_ind)**2))
    
    if len(rec.shape) == 2:
        rec  = r.reshape(rec.shape[0],rec.shape[1])
    elif len(rec.shape) == 3:
        rec  = r.reshape(rec.shape[0],rec.shape[1],rec.shape[2])
    else:
        raise Exception('Wrong shape of reconstruction!!!')
    return rsnr, rec

def ComputeRE(rec,target):
    """
    Compute relative error between the rec and target
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
       
    rec    = rec.reshape(np.size(rec))
    target = target.reshape(np.size(rec))
    rerror = np.sqrt(sum((target-rec)**2)) / np.sqrt(sum(target**2))
    
    return rerror



def AddAWGN(data, snr):
    """
       Add additive white Gaussian noise to data such that the SNR is snr
    """
    if len(data.size()) !=3:
        assert False, 'Please check the data shape!!!'
    
    # change the shape to [num_shots,nt,num_receiver]
    data1 = data
    dim = data1.size() 
    device = data1.device
    SNR = snr
    y_noisy = data1 + torch.randn(dim).to(device)*(torch.sqrt(torch.mean((data1.detach()**2).reshape(dim[0],-1),dim=1)/(10**(SNR/10)))).reshape(dim[0],1,1).repeat(1,dim[1],dim[2])
    
    # change the shape to [nt,num_shots,num_receiver]
                       
    # check the shape of y_noisy is equal to data or not
    if y_noisy.size() != data.size():
        assert False, 'Wrong shape of noisy data!!!'                 
  
    return y_noisy


def TVLoss(x):
    """Compute TV loss for an image x
        Args:
            x: image, torch.Variable of torch.Tensor
        Returns:
            tv loss
     """
    x      = x.float()
    dh     = torch.pow(x[:,1:] - x[:,:-1], 2)
    dw     = torch.pow(x[1:,:] - x[:-1,:], 2)
    tvloss = torch.sum(torch.pow(dh[:-1,:] + dw[:, :-1], 0.5)).float()

    return tvloss



def ATVLoss(x):    
    """Compute L1-based anisotropic TV loss for x

    Args:
        x: image, torch.Variable of torch.Tensor
    Returns:
           ATV loss
    """
    x        = x.float()
    dh       = x[:,1:] - x[:,:-1]
    dw       = x[1:,:] - x[:-1,:]
    atvloss  = torch.sum(torch.abs(dh[:-1,:]) + torch.abs(dw[:, :-1])).float()

    return atvloss

def updateinput(net_input_saved,noise,i,reg_noise_std,reg_noise_decayevery):
    """
    update the input of decoder
    """
    
    if reg_noise_decayevery !=0 and i % reg_noise_decayevery == 0:
        reg_noise_std *= 0.7
    net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        
    return net_input

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_num,input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [input_num, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params
    
def MSE(x, y):
    mse = np.sqrt(np.mean((x - y)**2))
    print(f"MSE = {mse}")
    return mse

def SSIM(x, y):
    k1 = 0.01
    k2 = 0.03
    vmax = np.max(x)
    vmin = np.min(x)
    μx = np.mean(x)
    μy = np.mean(y)
    σx = np.sqrt(np.mean((x-μx)**2))
    σy = np.sqrt(np.mean((y-μy)**2))
    σxy = np.mean((x-μx)*(y-μy))
    c1 = ( k1 * (vmax-vmin) )**2
    c2 = ( k2 * (vmax-vmin) )**2
    ssim = (2*μx*μy + c1) * (2*σxy + c2) / ((μx**2 + μy**2 + c1) * (σx**2 + σy**2 + c2))
#     print((2*μx*μy + c1), (μx**2 + μy**2 + c1),(2*σxy + c2), (σx**2 + σy**2 + c2))
    print(f"SSIM = {ssim}")
    return ssim

def PSNR(x, y):
    mse = np.sqrt(np.mean((x - y)**2))
    vmax = np.max(x)
    vmin = np.min(x)
    maxI = vmax - vmin
    psnr = 20.0 * np.log10(maxI) - 10.0 * np.log10(mse)
    print(f"PSNR = {psnr}")
    return psnr

def metrics(x, y):
    mse = MSE(x, y)
    ssim = SSIM(x, y)
    psnr = PSNR(x, y)    


class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, F_PEAK,size,
                 src,src_loc, rec_loc,rp_properties=None):
        super(Physics_deepwave, self).__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        self.F_PEAK = F_PEAK
        self.size = size
        rp_properties = rp_properties
    
    def forward(self, vp,vs,rho):
        out = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs,rho),
            self.dh, self.dt,
            source_amplitudes_y=self.src,
            source_amplitudes_x=self.src,
            source_locations_y=self.src_loc,
            source_locations_x=self.src_loc,
            receiver_locations_y=self.rec_loc,
            receiver_locations_x=self.rec_loc,
            pml_freq=self.F_PEAK
            )
        vx = out[15]
        vy = out[14]
        return vx.permute(0, 2, 1).unsqueeze(0),vy.permute(0, 2, 1).unsqueeze(0)

#   Source estimation 
def estimate_source_wavelet_elastic(vp_initial,vs_initial,rho_initial, obs_data, src_loc, rec_loc,rec_interp, inpa, Physics, DEVICE):
    """估计震源子波"""
    nr = inpa['nr']
    ns = inpa['ns']
    nt = inpa['nt']
    dh = inpa['dh']
    dt = inpa['dt']
    ns_per_shot = inpa['ns_per_shot']
    dtype = torch.float32
    xweight = torch.ones(nr, dtype=dtype, device=DEVICE)
    
    print(nr,nt,ns)
    
    stf_all =[]
   
   

    for s in range(ns):
        src_loc_new = src_loc[s,:,:].unsqueeze(0)
        
        src_amp = torch.zeros((1, ns_per_shot, nt), dtype=dtype, device=DEVICE)
        src_amp[0,0,0] = 1.0  # Dirac脉冲
        
        
        rec_loc_new = rec_loc[s,:,:].unsqueeze(0)
        d_obs_new = obs_data[s,:,:].to(DEVICE)
        #print(d_obs_new.shape)
	
        # 使用当前速度模型生成合成数据
    
        physics = Physics(dh, dt, inpa['fdom'],nt,src_amp,src_loc_new, rec_loc_new)
        
        taux_est = physics(vp_initial,vs_initial,rho_initial)  
	    #d_obs_vx = taux_est[0]
	    #d_obs_vy = taux_est[1]
        
        synthetic = rec_interp.receiver(taux_est[0].squeeze(0).cpu().permute(0, 2, 1)).permute(0, 2, 1).squeeze(0).to(DEVICE)
        #print(synthetic.shape)
        #rec_interp.receiver(d_obs_vy.squeeze().cpu()).transpose(1, 2).to(DEVICE)
        
        #d_syn.permute(1, 0).to(DEVICE)
        
        
        ntfft = 1 << (2*nt - 1).bit_length()
        G = torch.fft.rfft(synthetic * xweight.view(1,-1), n=ntfft, dim=0)
        D = torch.fft.rfft(d_obs_new * xweight.view(1,-1), n=ntfft, dim=0)

        num = torch.sum(torch.conj(G) * D, dim=1)
        den = torch.sum(torch.abs(G)**2, dim=1)
        eps = 1e-4 * torch.max(den)
        W = num / (den + eps)

        stf = torch.fft.irfft(W, n=ntfft, dim=0)[:nt]
        stf_all.append(stf)

    return torch.stack(stf_all, dim=0)



def encode_sources(obs_data_vx,obs_data_vy, source_locations, receiver_locations, source_functions,
                   num_super_sources, sources_per_super, source_distribution='uniform'):
    """
    将多个独立震源编码为超级震源
    
    参数:
    - obs_data: 观测数据 [114, 600, 456]
    - source_locations: 震源位置 [114, 1, 2] 
    - receiver_locations: 接收器位置 [114, 1143, 2]
    - source_functions: 震源函数 [114, 1, 600]
    - source_distribution: 震源分布方式 ('random' 或 'uniform')
    """
    
    ns, nt, nr = obs_data_vx.shape
    num_super_sources = num_super_sources
    sources_per_super = sources_per_super # 114 / 19 = 6
    
    # 1. 生成索引和分组策略
    if source_distribution == 'random':
        np.random.seed(42)
        indx = np.random.permutation(ns)
        # 随机分布：直接按排列后的顺序分组
        groups = []
        for i in range(num_super_sources):
            start_idx = i * sources_per_super
            end_idx = (i + 1) * sources_per_super
            groups.append(indx[start_idx:end_idx])
    else:
        # 均匀分布：使用等间隔的震源
        indx = np.linspace(0, ns-1, ns, dtype=int)
        groups = []
        for i in range(num_super_sources):
            # 选择等间隔的震源：i, i+19, i+38, ..., i+5*19
            group_indices = [i + j * num_super_sources for j in range(sources_per_super)]
            groups.append(group_indices)
    
    # 2. 编码观测数据 [114, 600, 456] -> [19, 600, 456]
    encoded_obs_data_vx = torch.zeros(num_super_sources, nt, nr)
    encoded_obs_data_vy = torch.zeros(num_super_sources, nt, nr)
    for i, group_indices in enumerate(groups):
        # 对组内的6个震源数据进行叠加
        encoded_obs_data_vx[i] = torch.sum(obs_data_vx[group_indices], dim=0)
        encoded_obs_data_vy[i] = torch.sum(obs_data_vy[group_indices], dim=0)
    
    # 3. 编码震源位置 [114, 1, 2] -> [19, 6, 2]
    encoded_source_locations = torch.zeros(num_super_sources, sources_per_super, 2)
    for i, group_indices in enumerate(groups):
        encoded_source_locations[i] = source_locations[group_indices].squeeze(1)
    
    # 4. 编码接收器位置 [114, 1143, 2] -> [19, 1143, 2]
    # 使用每个组第一个震源的接收器位置
    encoded_receiver_locations = torch.zeros(num_super_sources, receiver_locations.shape[1], 2)
    for i, group_indices in enumerate(groups):
        encoded_receiver_locations[i] = receiver_locations[group_indices[0]]
    
    # 5. 编码震源函数 [114, 1, 600] -> [19, 6, 600]
    encoded_source_functions = torch.zeros(num_super_sources, sources_per_super, nt)
    for i, group_indices in enumerate(groups):
        encoded_source_functions[i] = source_functions[group_indices].squeeze(1)
    
    return {
        'encoded_obs_data_vx': encoded_obs_data_vx,  # [19, 600, 456]
        'encoded_obs_data_vy': encoded_obs_data_vy,  # [19, 600, 456]
        'encoded_source_locations': encoded_source_locations,  # [19, 6, 2]
        'encoded_receiver_locations': encoded_receiver_locations,  # [19, 1143, 2]
        'encoded_source_functions': encoded_source_functions,  # [19, 6, 600]
        'groups': groups  # 分组信息
    }


def encode_sources_function(source_functions, source_distribution='uniform'):
    """
    将多个独立震源编码为超级震源
    
    参数:
    - obs_data: 观测数据 [114, 600, 456]
    - source_locations: 震源位置 [114, 1, 2] 
    - receiver_locations: 接收器位置 [114, 1143, 2]
    - source_functions: 震源函数 [114, 1, 600]
    - source_distribution: 震源分布方式 ('random' 或 'uniform')
    """
    
    ns, nt, nr = obs_data_vx.shape
    num_super_sources = 38
    sources_per_super = 3  # 114 / 19 = 6
    
    # 1. 生成索引和分组策略
    if source_distribution == 'random':
        np.random.seed(42)
        indx = np.random.permutation(ns)
        # 随机分布：直接按排列后的顺序分组
        groups = []
        for i in range(num_super_sources):
            start_idx = i * sources_per_super
            end_idx = (i + 1) * sources_per_super
            groups.append(indx[start_idx:end_idx])
    else:
        # 均匀分布：使用等间隔的震源
        indx = np.linspace(0, ns-1, ns, dtype=int)
        groups = []
        for i in range(num_super_sources):
            # 选择等间隔的震源：i, i+19, i+38, ..., i+5*19
            group_indices = [i + j * num_super_sources for j in range(sources_per_super)]
            groups.append(group_indices)
    
    # 5. 编码震源函数 [114, 1, 600] -> [19, 6, 600]
    encoded_source_functions = torch.zeros(num_super_sources, sources_per_super, nt)
    for i, group_indices in enumerate(groups):
        encoded_source_functions[i] = source_functions[group_indices].squeeze(1)
    
    return  encoded_source_functions

def encode_source_functions_only(source_functions, groups):
    """
    source_functions: [ns, 1, nt]
    groups: list of index groups
    """
    num_super_sources = len(groups)
    sources_per_super = len(groups[0])
    nt = source_functions.shape[-1]

    encoded_source_functions = torch.zeros(num_super_sources, sources_per_super, nt)

    for i, group_indices in enumerate(groups):
        encoded_source_functions[i] = source_functions[group_indices].squeeze(1)

    return encoded_source_functions

def plot_observed_data(d_obs_vx, d_obs_vy, p, device, figsize=(14, 4), save_path=None):
    """
    绘制观测数据的地震记录图
    
    Parameters:
    -----------
    d_obs_vx : torch.Tensor
        x方向观测数据，shape通常为 (1, time_samples, receivers) 或 (time_samples, receivers)
    d_obs_vy : torch.Tensor
        y方向观测数据，shape通常为 (1, time_samples, receivers) 或 (time_samples, receivers)
    device : torch.device
        设备（CPU或GPU）
    figsize : tuple, optional
        图像大小，默认为 (14, 4)
    save_path : str, optional
        保存图像的路径，如果为None则不保存
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图像对象
    ax : numpy.ndarray
        坐标轴对象数组
    """
    import matplotlib.pyplot as plt
    import torch
    
    # 移除多余的维度（如果存在）
    d_obs_vx_plot = d_obs_vx.squeeze(0) if d_obs_vx.dim() > 3 else d_obs_vx
    d_obs_vy_plot = d_obs_vy.squeeze(0) if d_obs_vy.dim() > 3 else d_obs_vy
    
    print(f"Data shape - Vx: {d_obs_vx_plot.shape}, Vy: {d_obs_vy_plot.shape}")
    
    # 计算百分位数用于颜色映射范围
    vmin, vmax = torch.quantile(d_obs_vx_plot[p],
                                torch.tensor([0.01, 0.99]).to(device))
    vsmin, vsmax = torch.quantile(d_obs_vy_plot[p],
                                  torch.tensor([0.01, 0.99]).to(device))
    
    # 创建图像
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    # 绘制Vx分量
    im1 = ax[0].imshow(d_obs_vx_plot[p].cpu().detach().numpy(), 
                       aspect='auto', cmap='gray', 
                       vmin=vmin, vmax=vmax)
    ax[0].set_xlabel("Receiver")
    ax[0].set_ylabel("Time sample")
    plt.colorbar(im1, ax=ax[0], label='Amplitude')
    
    # 绘制Vy分量
    im2 = ax[1].imshow(d_obs_vy_plot[p].cpu().detach().numpy(), 
                       aspect='auto', cmap='gray',
                       vmin=vsmin, vmax=vsmax)
    ax[1].set_xlabel("Receiver")
    # ax[1].set_ylabel("Time sample")  # 注释掉以保持与原代码一致
    plt.colorbar(im2, ax=ax[1], label='Amplitude')
    
    # 设置标题
    ax[0].set_title("Observed Data - Vx Component")
    ax[1].set_title("Observed Data - Vy Component")
    
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    # 保存图像（如果需要）
    if save_path is not None:
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def gaussian_smooth_once(vp_true, vs_true, rho_true, sigma, device):
    vp_np = vp_true.detach().cpu().numpy()
    vs_np = vs_true.detach().cpu().numpy()
    rho_np = rho_true.detach().cpu().numpy()

    vp_s = scipy.ndimage.gaussian_filter(vp_np, sigma=sigma)
    vs_s = scipy.ndimage.gaussian_filter(vs_np, sigma=sigma)
    rho_s = scipy.ndimage.gaussian_filter(rho_np, sigma=sigma)

    return (
        torch.tensor(vp_s, dtype=vp_true.dtype, device=device),
        torch.tensor(vs_s, dtype=vs_true.dtype, device=device),
        torch.tensor(rho_s, dtype=rho_true.dtype, device=device),
    )


def gaussian_smooth_n_times(vp_true, vs_true, rho_true, sigma, times, device):
    # ⚠️ 关键：从 true 开始，但每次都用“上一次结果”
    vp_np = vp_true.detach().cpu().numpy()
    vs_np = vs_true.detach().cpu().numpy()
    rho_np = rho_true.detach().cpu().numpy()

    for _ in range(times):
        vp_np = scipy.ndimage.gaussian_filter(vp_np, sigma=sigma)
        vs_np = scipy.ndimage.gaussian_filter(vs_np, sigma=sigma)
        rho_np = scipy.ndimage.gaussian_filter(rho_np, sigma=sigma)

    return (
        torch.tensor(vp_np, dtype=vp_true.dtype, device=device),
        torch.tensor(vs_np, dtype=vs_true.dtype, device=device),
        torch.tensor(rho_np, dtype=rho_true.dtype, device=device),
    )

def plot_wavelet_spectrum_combined(src_true, src_est, dt, p=2, q=200, q_fft=2):
                                   #save_path='./combined_wavelet_spectrum.pdf'):
    """
    绘制时域波形和频谱分析组合图
    
    参数:
    ----------
    src_true : array-like
        真实信号源，形状为 [n_sources, 1, n_samples]
    src_est : array-like
        估计信号源，形状为 [n_sources, 1, n_samples]
    p : int, 默认=10
        时域波形分析的第p个信号源索引
    q : int, 默认=300
        时域波形显示的前q个采样点
    q_fft : int, 默认=20
        频谱分析的第q_fft个信号源索引
    save_path : str, 默认='./combined_wavelet_spectrum.pdf'
        保存图片的路径
    """
    # 提取时域波形数据
    src_wavelet = src_true[p, 0, :].squeeze()
    src_est_wavelet = src_est[p, 0, :].squeeze()
    
    # 提取频谱分析数据
    wavelet_org = src_true[q_fft, 0, :]
    trace_org = wavelet_org.cpu().numpy() if hasattr(wavelet_org, 'cpu') else wavelet_org
    
    wavelet_est = src_est[q_fft, 0, :]
    trace_est = wavelet_est.cpu().numpy() if hasattr(wavelet_est, 'cpu') else wavelet_est
    
    # 计算频谱
    NT = src_est.shape[2]
    DT = dt  # 采样间隔，根据实际情况调整
    
    f = np.fft.fftfreq(NT, DT)
    
    wave_fft_org = np.fft.fft(trace_org)
    wavelet_fft_shifted_org = np.fft.fftshift(wave_fft_org)
    freqs_shifted_org = np.fft.fftshift(f)
    
    wave_fft_est = np.fft.fft(trace_est)
    wavelet_fft_shifted_est = np.fft.fftshift(wave_fft_est)
    freqs_shifted_est = np.fft.fftshift(f)
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个子图：时域波形
    ax1.plot(src_wavelet[0:q], color='blue', linestyle=':', linewidth=1.5, label='True')
    ax1.plot(src_est_wavelet[0:q], color='green', linestyle='-.', linewidth=1.5, label='Estimated')
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Amplitude', fontsize=14)
    ax1.legend(loc='upper right', fontsize=12, frameon=True, framealpha=1, 
               edgecolor='black', fancybox=False)
    ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 第二个子图：频谱分析
    ax2.plot(freqs_shifted_org, np.abs(wavelet_fft_shifted_org), color='blue', 
             linestyle='-', linewidth=2.0, label='True source')
    ax2.plot(freqs_shifted_est, np.abs(wavelet_fft_shifted_est), color='orange', 
             linestyle=':', linewidth=2.0, label='Estimated')
    ax2.set_xlabel('Frequency (Hz)', fontsize=16)
    ax2.set_ylabel('Amplitude', fontsize=16)
    ax2.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax2.set_xticklabels([0, 5, 10, 15, 20, 25, 30], fontsize=15)
    ax2.legend(fontsize='x-large')
    ax2.set_xlim(0, 30)
    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    # plt.savefig(save_path, dpi=300, format='pdf', 
    #             bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return fig, (ax1, ax2)

# 使用示例
if __name__ == "__main__":
    # 假设你已有这些变量
    # d_obs_vx, d_obs_vy, DEVICE 已定义
    
    # 方式1：基本使用
    fig, ax = plot_observed_data(d_obs_vx, d_obs_vy, DEVICE)
    
    # 方式2：自定义图像大小
    fig, ax = plot_observed_data(d_obs_vx, d_obs_vy, DEVICE, figsize=(16, 5))
    
    # 方式3：保存图像
    fig, ax = plot_observed_data(d_obs_vx, d_obs_vy, DEVICE, 
                                 save_path="./output/observed_data.jpg")
    
    # 方式4：使用所有参数
    fig, ax = plot_observed_data(d_obs_vx, d_obs_vy, DEVICE, 
                                 figsize=(14, 4),
                                 save_path="./results/obs_gather.jpg")
