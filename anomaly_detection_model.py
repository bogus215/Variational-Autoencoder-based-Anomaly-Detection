#%% library
from loader import loader
from paper_model import VAE,AE
from easydict import EasyDict as edict
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
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
print('VAE novelty detection confusion matrix',confusion_matrix(novelty_detection,label_list))

#%% Reconstruction이 효과적으로 되지 않았던 이미지 데이터 시각화 - vae
feature_list_low = []
feature_list_high = []
for idx, (feature, label) in enumerate(config.loader.test_iter):
    feature = feature.cuda(config.gpu_device)
    probability = config.VAE.reconstruction_probability(feature).detach()
    feature_list_low.append(feature[probability <= threshold]) # Reconstruction probability가 중위수가 낮았던 feature 추출
    feature_list_high.append(feature[probability > threshold]) # Reconstruction probability가 중위수가 낮았던 feature 추출

feature_list_low , feature_list_high = torch.cat(feature_list_low) , torch.cat(feature_list_high)
feature_list_low = feature_list_low[np.random.choice(feature_list_low.__len__(),10)] # 복원이 잘 되지 않았던 이미지 중 10개 추출
feature_list_high = feature_list_high[np.random.choice(feature_list_high.__len__(),10)] # 복원이 잘 되지 않았던 이미지 중 10개 추출

reconstrucion_visual_vae(feature_list_low,config,title= 'vae_reconstruction_with_low_probability')
reconstrucion_visual_vae(feature_list_high,config,title= 'vae_reconstruction_with_high_probability')



#%% novelty detection - ae
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
print('AE novelty detection performance',confusion_matrix(novelty_detection,label_list))


#%% Reconstruction이 효과적으로 되지 않았던 이미지 데이터 시각화 - ae
feature_list_low = []
feature_list_high = []
for idx, (feature, label) in enumerate(config.loader.test_iter):
    feature = feature.cuda(config.gpu_device)
    error = torch.mean(torch.square(config.AE(feature) - feature), axis=[2, 3, 1])
    feature_list_low.append(feature[error > threshold]) # Reconstruction error가 중위수가 높았던 feature 추출
    feature_list_high.append(feature[error <= threshold]) # Reconstruction error가 중위수가 낮았던 feature 추출

feature_list_low , feature_list_high= torch.cat(feature_list_low) , torch.cat(feature_list_high)
feature_list_low = feature_list_low[np.random.choice(feature_list_low.__len__(),10)] # 복원이 잘 되지 않았던 이미지 중 10개 추출
feature_list_high = feature_list_high[np.random.choice(feature_list_high.__len__(),10)] # 복원이 잘 되지 않았던 이미지 중 10개 추출
reconstrucion_visual_ae(feature_list_low,config,title= 'ae_reconstruction_with_high_error')
reconstrucion_visual_ae(feature_list_high,config,title= 'ae_reconstruction_with_low_error')


def reconstrucion_visual_vae(feature,config,title):
    fig = plt.figure(figsize=(10, 8))
    plt.title(title)
    images = config.VAE(feature.squeeze(1).view(10,-1))[0].detach().cpu().numpy().reshape(10, 28, 28)
    first_row = np.concatenate(images[:5,:,:],axis=1)
    second_row = np.concatenate(images[5:,:,:],axis=1)
    plt.imshow(np.concatenate([first_row,second_row]))
    plt.savefig(f'./img/{title}.png')
    plt.show()

def reconstrucion_visual_ae(feature,config,title):
    fig = plt.figure(figsize=(10, 8))
    plt.title(title)
    images = config.AE(feature.squeeze(1).view(10,-1)).detach().cpu().numpy().reshape(10, 28, 28)
    first_row = np.concatenate(images[:5,:,:],axis=1)
    second_row = np.concatenate(images[5:,:,:],axis=1)
    plt.imshow(np.concatenate([first_row,second_row]))
    plt.savefig(f'./img/{title}.png')
    plt.show()
