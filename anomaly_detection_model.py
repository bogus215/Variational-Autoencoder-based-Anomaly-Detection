#%% library
from loader import loader
from paper_model import VAE,AE
from easydict import EasyDict as edict
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score , f1_score, precision_score, recall_score, confusion_matrix

#%% utils
def return_result(y_pred, y_true):
    result = {'accuracy': accuracy_score(y_true, y_pred),
              'f1': f1_score(y_true, y_pred,average='weighted'),
              'precision': precision_score(y_true, y_pred,average='weighted'),
              'recall': recall_score(y_true, y_pred,average='weighted')}
    return result



#%% config
config = edict()
config.gpu_device = 0
config.batch_size = 128
config.abnormal_class = 0
config.loader = loader(config)
config.parameter_path = "D:/2020-2/비즈니스애널리틱스/논문리뷰/Variational Autoencoder based Anomaly Detection/Variational-Autoencoder-based-Anomaly-Detection/parameter"

#%% model load - VAE AND AE
config.VAE = VAE(input_size=28*28).cuda(config.gpu_device)
parameter = torch.load(os.path.join(config.parameter_path,'best_parameter_Abnormal_class_0_vae.pth'))
config.VAE.load_state_dict(parameter)

config.AE = AE(input_size=28*28).cuda(config.gpu_device)
parameter = torch.load(os.path.join(config.parameter_path,'best_parameter_Abnormal_class_0_ae.pth'))
config.AE.load_state_dict(parameter)

#%% novelty detection - vae
config.VAE.eval()
Reconstruction_prob = []
label_list = []
for idx, (feature, label) in enumerate(config.loader.test_iter):
    feature = feature.cuda(config.gpu_device)
    probability = config.VAE.reconstruction_probability(feature).detach()
    Reconstruction_prob.append(probability)
    label_list.append(label)
Reconstruction_prob = torch.cat(Reconstruction_prob).cpu().numpy()
threshold = np.quantile(Reconstruction_prob,q=0.5)
novelty_detection = np.where(Reconstruction_prob <= threshold,1,0)
label_list = np.where(torch.cat(label_list).numpy() ==0 , 1 , 0)
print('VAE novelty detection performance',return_result(novelty_detection,label_list))


#%% novelty detection - vae
config.AE.eval()
Reconstruction_error = []
label_list = []
for idx, (feature, label) in enumerate(config.loader.test_iter):
    feature = feature.cuda(config.gpu_device)
    error = torch.mean(torch.square(config.AE(feature) - feature), axis=[2,3,1])
    Reconstruction_error.append(error)
    label_list.append(label)
Reconstruction_error = torch.cat(Reconstruction_error).detach().cpu().numpy()
threshold = np.quantile(Reconstruction_error,q=0.5)
novelty_detection = np.where(Reconstruction_error < threshold,0,1)
label_list = np.where(torch.cat(label_list).numpy() ==0 , 1 , 0)
print('AE novelty detection performance',return_result(novelty_detection,label_list))
