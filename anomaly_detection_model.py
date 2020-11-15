#%% library
from loader import loader
from paper_model import VAE,AE
from easydict import EasyDict as edict
import torch
import os
from sklearn.metrics import accuracy_score , f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

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

#%% novelty detection
config.VAE.eval()
Reconstruction_prob = []
label_list = []
for idx, (feature, label) in enumerate(config.loader.test_iter):
    feature = feature.cuda(config.gpu_device)
    probability = config.VAE.reconstruction_probability(feature).detach().mean(axis=0)
    Reconstruction_prob.append(probability)
    label_list.append(label)
Reconstruction_prob = torch.cat(Reconstruction_prob).cpu().numpy()
threshold = np.quantile(Reconstruction_prob,q=0.1)
novelty_detection = np.where(Reconstruction_prob <= threshold,1,0)
label_list = np.where(torch.cat(label_list).numpy() ==0 , 1 , 0)




def show_visual_progress(config, rows=5, title=None):

    fig = plt.figure(figsize=(10, 8))
    if title:
        plt.title(title)

    image_rows = []
    for idx, (feature, label) in enumerate(config.loader.test_iter):
        if rows == idx:
            break
        feature = feature.cuda(config.gpu_device)
        images = feature.detach().cpu().numpy().reshape(feature.size(0), 28, 28)
        images_idxs = [list(label.numpy()).index(x) for x in range(1,10)]
        combined_images = np.concatenate([images[x].reshape(28, 28) for x in images_idxs],
                                         1)
        image_rows.append(combined_images)

    plt.imshow(np.concatenate(image_rows))
    plt.savefig('./img/' + title + '.png', dpi=300)
