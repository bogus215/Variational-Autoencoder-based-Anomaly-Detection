# %% library
from loader import loader
import argparse
from paper_model import VAE
import numpy as np
import torch
import torch.optim as optim
from pytorchtools import EarlyStopping
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
import random
import matplotlib.pyplot as plt


# %% Train
def train(args):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    optimizer = optim.Adam(args.model.parameters(), args.learning_rate)
    best_MSE = np.inf
    start = time.time()

    writer = SummaryWriter(f'./runs/{args.experiment}')
    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.experiment}.pth')

    for e in range(args.epoch):
        print("\n===> epoch %d" % e)

        total_loss = 0

        for i, batch in enumerate(tqdm(args.loader.train_iter, desc='train')):

            feature = batch[0].cuda(args.gpu_device)
            optimizer.zero_grad()
            args.model.train()
            x_mean, x_logvar, z_mean, z_logvar = args.model(feature)

            kl_divergence = -0.5 * torch.sum(1 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar), axis=1)
            log_recon_likelihood = -0.5 * (torch.sum(torch.square(feature-x_mean) * torch.exp(-x_logvar) , axis = [2,3,1]) + torch.sum(x_logvar ,axis=[2,3,1]) + 784 * np.log(2*np.pi))
            loss = torch.mean(kl_divergence - log_recon_likelihood)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % args.printevery == 0:

                with torch.no_grad():

                    args.model.eval()
                    val_loss = 0

                    for s, val_batch in enumerate(tqdm(args.loader.valid_iter, desc='valid')):
                        feature = val_batch[0].cuda(args.gpu_device)
                        # target = val_batch[1].cuda(args.gpu_device)
                        x_mean, x_logvar, z_mean, z_logvar = args.model(feature)

                        kl_divergence = -0.5 * torch.sum(1 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar),axis=1)
                        log_recon_likelihood = -0.5 * (torch.sum(torch.square(feature - x_mean) * torch.exp(-x_logvar),axis=[2, 3, 1]) + torch.sum(x_logvar, axis=[2, 3,1]) +
                                                       784 * np.log(2 * np.pi))
                        loss = torch.mean(kl_divergence - log_recon_likelihood)
                        val_loss += loss.item()

                if best_MSE > (val_loss / len(args.loader.valid_iter)):
                    best_MSE = (val_loss / len(args.loader.valid_iter))
                    torch.save(args.model.state_dict(), f'./parameter/best_parameter_{args.experiment}.pth')

                iters = (e) * (len(args.loader.train_iter)) + i
                avg_loss = total_loss / args.printevery

                writer.add_scalar('train_loss', avg_loss, iters + 1)
                writer.add_scalar('valid_loss', val_loss / len(args.loader.valid_iter), iters + 1)
                total_loss = 0
                show_visual_progress(args, rows=5, title=f'{args.experiment}_{iters}')
                plt.close('all')
                early_stopping(val_loss / len(args.loader.valid_iter), args.model)

                if early_stopping.early_stop:
                    print('Early stopping')
                    break

        if early_stopping.early_stop:
            print('Early stopping')
            break




def show_visual_progress(args, rows=5, title=None):

    fig = plt.figure(figsize=(10, 8))
    if title:
        plt.title(title)

    image_rows = []
    for idx, (feature, label) in enumerate(args.loader.test_iter):
        if rows == idx:
            break
        feature = feature.cuda(args.gpu_device)
        images = args.model(feature)[0].detach().cpu().numpy().reshape(feature.size(0), 28, 28)
        images_idxs = [list(label.numpy()).index(x) for x in range(10)]
        combined_images = np.concatenate([images[x].reshape(28, 28) for x in images_idxs],
                                         1)
        image_rows.append(combined_images)

    plt.imshow(np.concatenate(image_rows))
    plt.savefig('./img/' + title + '.png', dpi=300)



# %% main
def main():
    parser = argparse.ArgumentParser(description="-----[#]-----")

    # Model
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument('--input_dimension', type=int, default=1, help='이미지 가로 차원 수')
    parser.add_argument('--latent_dimension', type=int, default=25, help='latent variable dimension')

    # Data and train
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training [default: 128]')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--printevery', default=100, type=int, help='log , print every % iteration')
    parser.add_argument('--experiment', type=str, default='Abnormal_class_0_vae', help='experiment name')
    parser.add_argument('--abnormal_class', type=int, default=0, help='abnormal class')


    args = parser.parse_args()
    args.loader = loader(args)
    args.model = VAE(input_size=28*28).cuda(args.gpu_device)

    gc.collect()
    train(args)


# %% run
if __name__ == "__main__":
    main()