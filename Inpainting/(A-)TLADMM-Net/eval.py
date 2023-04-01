import torch
import numpy as np
from os.path import dirname, abspath
import os
import matplotlib.pyplot as plt
from PIL import Image
import useful_utils
import generating
import torch.utils.data as Data
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# load D that was calculated with train set
dict_path = "./data/data_8x8_N_100000_atoms_256.npz"
save_path = "./figures/inpainting/figs_for_paper/"
npzfile = np.load(dict_path, allow_pickle=True)
D, mean_avg, mean_std = (npzfile["D"].astype(np.float32), npzfile["avg_mean"], npzfile["avg_std"])
NUM_ATOMS = np.shape(D)[1]

T = 20
lambd = 0.1
RATIO = 0.5
data_path = "./data/set11"
image_list = sorted(os.listdir(data_path))
PATCH_SIZE = 8
n = PATCH_SIZE ** 2
OVERLAP = True
BATCH_SIZE = 256
cudaopt = True

# Load (A-)TLADMM Model
model_ours_path = f"./saved_models_inpainting/A_TLADMM_layers_{T}_ratios_{RATIO}"
model_ours = torch.load(model_ours_path)
model_ours.eval()
params = list(model_ours.parameters())
# for param in params:
# 	print(param)

def infer_ours(corrupt_patches, mask_patches, D, model):
    NUM_PATCHES = np.shape(corrupt_patches)[1]
    data_for_loader = generating.SimulatedDataNoisedDict(
        y=torch.from_numpy(corrupt_patches),
        D_noised=torch.from_numpy(mask_patches),
        x=torch.zeros(NUM_ATOMS, NUM_PATCHES),
    )
    data_loader = Data.DataLoader(
        dataset=data_for_loader, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )
    count = 0
    recon_patches = np.zeros_like(corrupt_patches)
    with torch.no_grad():
        for (b_y, b_M, b_x) in data_loader:
            if cudaopt:
                b_y, b_M, b_x = b_y.cuda(), b_M.cuda(), b_x.cuda()
            x_hat = model(b_y, b_M).T
            x_hat = (x_hat.data).cpu().numpy()
            if (count + 1) * BATCH_SIZE <= NUM_PATCHES:
                recon_patches[:, count * BATCH_SIZE : (count + 1) * BATCH_SIZE] = D @ x_hat
            else:
                AAA = D @ x_hat
                if len(AAA.shape) == 1:
                    recon_patches[:, count * BATCH_SIZE :] = AAA.reshape(-1,1)
                else:
                    recon_patches[:, count * BATCH_SIZE :] = AAA
            count += 1

    return recon_patches


def infer_ista(corrupt_patches, mask_patches, D, lambd, T, fista_flag=False, normalize_atoms=False):

    if fista_flag:
        # print("Computing Fista...")
        x_hat = generating.fista_inpainting(
            y=torch.from_numpy(corrupt_patches),
            D=torch.from_numpy(D),
            M=torch.from_numpy(mask_patches),
            lambd=lambd,
            L=None,
            max_itr=T,
            same_L=True,
        )
    else:
        # print("Computing Ista...")
        x_hat = generating.ista_inpainting(
            y=torch.from_numpy(corrupt_patches),
            D=torch.from_numpy(D),
            M=torch.from_numpy(mask_patches),
            lambd=lambd,
            L=None,
            max_itr=T,
            same_L=True,
        )

    x_hat = x_hat.data.cpu().numpy()
    recon_patches = D @ x_hat
    return recon_patches


def patches2image(recon_patches, name2print="ISTA"):
    # Un-normalization
    recon_patches *= np.expand_dims(mean_std, axis=-1)
    recon_patches += np.expand_dims(mean_avg, axis=-1)
    # Image from patches
    recon_image = useful_utils.patches_to_image(recon_patches, H, W, OVERLAP)
    recon_psnr = useful_utils.compute_psnr(image, recon_image)
    print("Corrupt PSNR = %.3f, %s PSNR: %.3f" % (corrupt_psnr, name2print, recon_psnr))
    return recon_image, recon_psnr

Set11_NRMSE = 0
Set11_PSNR = 0
for i in range(len(image_list)):
    print(image_list[i])
    image = np.array(Image.open(data_path + "/" + image_list[i])) / 255.0
    H, W = image.shape[:2]
    mask = np.where(np.random.rand(H, W) < RATIO, 0, 1)
    # print(mask)
    # Extract corrupted patches
    corrupt_patches = useful_utils.image_to_patches(image * mask, PATCH_SIZE, OVERLAP).astype(
        np.float32
    )
    # Compute PSNR
    corrupt_psnr = useful_utils.compute_psnr(image, image * mask)
    N = np.shape(corrupt_patches)[1]
    # Mask for every patch
    mask_patches = useful_utils.image_to_patches(mask, PATCH_SIZE, OVERLAP).astype(np.float32)
    # Normalize patches
    patch_sum = np.sum(corrupt_patches, axis=-1)
    mask_sum = np.sum(mask_patches, axis=-1)
    mean_patch = np.expand_dims(patch_sum / mask_sum, axis=-1)
    mask_patches = (
        (np.expand_dims(mask_patches.T, axis=-1) * np.eye(n)).transpose(1, 2, 0).astype(np.float32)
    )
    corrupt_patches -= np.expand_dims(mean_avg, axis=-1)
    corrupt_patches /= np.expand_dims(mean_std, axis=-1)

    # ISTA Inference
    ista_recon_patches = infer_ista(
        corrupt_patches, mask_patches, D, lambd, T + 1, fista_flag=False
    )
    ista_image, ista_psnr = patches2image(ista_recon_patches, name2print="ISTA")

    # FISTA Inference
    fista_recon_patches = infer_ista(
        corrupt_patches, mask_patches, D, lambd, T + 1, fista_flag=True
    )
    fista_image, fista_psnr = patches2image(fista_recon_patches, name2print="FISTA")

    # Ada-LISTA Inference
    ours_recon_patches = infer_ours(corrupt_patches, mask_patches, D, model=model_ours)
    ours_image, ours_psnr = patches2image(ours_recon_patches, name2print="A-TLADMM")

    Set11_PSNR = Set11_PSNR + ours_psnr
    print(image_list[i] + " NRMSE: %.4f" % (torch.norm(torch.Tensor(ours_image) - torch.Tensor(image), p="fro")/torch.norm(torch.Tensor(image), p="fro")))
    Set11_NRMSE = Set11_NRMSE + torch.norm(torch.Tensor(ours_image) - torch.Tensor(image), p="fro")/torch.norm(torch.Tensor(image), p="fro")
    # Plot figures
    cv2.imwrite("./figures/inpainting/" + image_list[i].split(".")[0] + ".png", ours_image * 255)

Set11_NRMSE = Set11_NRMSE / len(image_list)
Set11_PSNR = Set11_PSNR / len(image_list)
print('NRMSE: %.4f, PSNR: %.4f ' % (Set11_NRMSE, Set11_PSNR))