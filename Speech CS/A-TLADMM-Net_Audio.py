import torch
import torch.nn as nn
import torchvision
import numpy as np
from datasets import MYSPEECHCOMMANDS, Timit
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import os
import lmdb
import pickle
from torch.nn import init
import torch.nn.functional as F 
from datetime import datetime
import random
import torchaudio
import librosa
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
parser = ArgumentParser(description = "A-TLADMM-Net")
parser.add_argument("--GPU_list", type = str, default = '3', help = "GPU index")
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument("--epoch_num", type = int, default = 100, help = "The number of epochs")
parser.add_argument("--data_path", type = str, default = "./data/timit_200_800_8000_orth", help = "The path of processed dataset")
parser.add_argument("--batch_size", type = int, default = 128, help = "The bach size for each iteration")
parser.add_argument("--layers_num", type = int, default = 10, help = "The number of layers")
parser.add_argument("--learning_rate", type = float, default = 1e-4, help = "learning rate")
parser.add_argument("--model_dir", type = str, default = "model", help = "the path of saved models")
parser.add_argument("--sample_rate", type = float, default = 8000, help = "sample rate")
parser.add_argument("--dataset", type = str, default = "speechcommands", help = "dataset")
parser.add_argument("--alg", type = str, default = "A-TLADMM-Net", help = "algorithm name")
args = parser.parse_args()

start_epoch = args.start_epoch
gpu_list = args.GPU_list
epochs = args.epoch_num
data_path = args.data_path
batch_size = args.batch_size
layers = args.layers_num
learning_rate = args.learning_rate
if args.dataset == "speechcommands":
    MAX_LENGTH = args.sample_rate
else:
    MAX_LENGTH = int(60000 * (args.sample_rate / 8000))
AMBIENT_DIM = 800

TEST_SAMPLE_INDICES_TO_SAVE = [10, 51, 201, 103, 1]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

with open(os.path.join(data_path, 'measurement_matrix.p'), 'rb') as fd:
    meaturement_matrix = pickle.load(fd)
Phi = meaturement_matrix
print('The CS ratio is:', Phi.shape[0]/AMBIENT_DIM)

def split_into_windows(t: torch.Tensor, num_windows: int = 10) -> torch.Tensor:
    return [x for x in t.reshape(num_windows, -1)]

def date_fname():
    uniq_filename = (
        str(datetime.now().date()) + "_" + str(datetime.now().time()).replace(":", ".")
    )
    return uniq_filename
def save_spectrogram(wav_path):
    spec_fn = torchaudio.transforms.Spectrogram(n_fft=1024)
    sig, sr = librosa.load(wav_path)
    mels = spec_fn(torch.tensor(sig))

    ax = plt.subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(mels, ref=np.max), ax=ax, sr=sr, y_axis='log', x_axis='time')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] #Linux Biolinum
    font2 = {'family' : 'Times New Roman', #Linux Biolinum
            'weight' : 'normal',
            'size' : 30,
            }
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Time', font2)
    plt.ylabel('Hz', font2)
    plt.savefig(f'{wav_path}.png', dpi=200, bbox_inches='tight')

    print(wav_path)

def save_examples(model, eval_loader, epoch, algo="admm_classic", device="cpu"):
    folder = f"results_speech_admm/{algo}_{date_fname()}_epoch.{epoch}"
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(e)
            raise IOError((f"Failed to create recursive directories: {folder}"))
    idxes = TEST_SAMPLE_INDICES_TO_SAVE
    wavs = [eval_loader.dataset[i][0] for i in idxes]

    segments = [
        torch.stack(
            split_into_windows(
                w, num_windows=int(MAX_LENGTH / AMBIENT_DIM)
            )
        ).to(device)
        for w in wavs
    ]

    segments = [s[s.sum(dim=-1) != 0] for s in segments]
    measurements = [model.measure_x(s.cuda()) for s in segments]
    reconstructed = []
    for m, s in zip(measurements, segments):
        x, y, Wloss = model(m, s) 
        reconstructed.append(x[-1])
    reconstructed = [r.reshape(-1).detach().cpu() for r in reconstructed]
    for idx, (org, rec) in enumerate(zip(wavs, reconstructed)):
        org = org[org != 0.0].unsqueeze(0)
        rec = rec.unsqueeze(0)
        org_path = os.path.join(folder, f"original_{idx}_epoch_{epoch}.wav")
        rec_path = os.path.join(folder, f"reconstructed_{idx}_epoch_{epoch}.wav")
        torchaudio.save(
            org_path,
            org,
            args.sample_rate,
        )
        torchaudio.save(
            rec_path,
            rec,
            args.sample_rate,
        )
        save_spectrogram(org_path)
        save_spectrogram(rec_path)
    return None

class A_ELADMMNet(torch.nn.Module):
    def __init__(self, LayerNo, Phi):
        super(A_ELADMMNet, self).__init__()
        self.LayerNo = LayerNo
        self.W = nn.Parameter(Phi)
        self.conv_size = 32
        self.beta1 = nn.Parameter(torch.ones(self.LayerNo))
        self.beta2 = nn.Parameter(torch.ones(self.LayerNo))
        self.h = nn.Parameter(torch.ones(self.LayerNo))
        self.soft_thr = nn.Parameter(0.01*torch.ones(self.LayerNo))
        self.thetax = nn.Parameter(0.01*torch.ones(self.LayerNo))
        self.thetaz = nn.Parameter(0.01*torch.ones(self.LayerNo))
        self.thetaL = nn.Parameter(0.01*torch.ones(self.LayerNo))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3))) 
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3))) 
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3))) 
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3))) 


    def self_active(self, x, thershold):
        s = torch.abs(thershold)
        return torch.mul(torch.sign(x), F.relu(torch.abs(x) - s))

    def measure_x(self, x):
        y = torch.einsum("ma,ba->bm", self.W, x)
        n = 1e-4*torch.randn_like(y)
        y = y+n
        return y

    def forward(self, y, x_original):
        y = y.view(y.size(0), -1)
        PhiTPhi = torch.mm(self.W.t(), self.W)
        PhiTb = torch.mm(y, self.W)

        min_x = torch.min(x_original)
        max_x = torch.max(x_original)
        
        X0 = torch.zeros((y.size(0), 800)).cuda()
        Z0 = torch.zeros((y.size(0), 800)).cuda()
        L0 = torch.zeros((y.size(0), 800)).cuda()
        
        X = list()
        Z = list()
        L = list()
        layers_sym = list()
        Wloss = torch.mm(self.W, self.W.t()) - torch.eye(self.W.shape[0], self.W.shape[0]).cuda()
        
        # 0.5*\|y-phix\|_2^2 + lambda*\|Ax\|_1 -----> 0.5*\|y-phix\|_2^2 + lambda*\|Az\|_1   z = x
        for i in range(self.LayerNo):
            if i == 0:
                # update X
                grad_x = PhiTb - torch.mm(X0, PhiTPhi) + (self.beta1[i] * (Z0 - X0) - L0)
                X.append(X0 + self.h[i] * grad_x)
                # update Z
                grad_z = L0 + self.beta2[i] * (X[-1] - Z0)
                z_input = (Z0 + self.h[i] * grad_z).view(-1, 1, 20, 40)
                x = F.conv2d(z_input, self.conv1_forward, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward, padding=1)
                x = F.conv2d(self.self_active(x_forward, self.soft_thr[i]), self.conv1_backward, padding=1)
                x = F.relu(x)
                x_backward = F.conv2d(x, self.conv2_backward, padding=1)
                Z.append(x_backward.view(-1, 800))
                # update L
                L.append(L0 + self.h[i] * self.beta1[i]*(X[-1] - Z[-1]))
                # A^TA
                x = F.conv2d(x_forward, self.conv1_backward, padding=1)
                x = F.relu(x)
                x_sym = F.conv2d(x, self.conv2_backward, padding=1)
                layers_sym.append(x_sym - z_input)
            
            elif i == 1:
                # update X
                hatx = X[-1] + self.thetax[i] * (X[-1] - X0) # acceleration
                grad_x = PhiTb - torch.mm(hatx, PhiTPhi) + (self.beta1[i] * (Z0 - hatx) - L0)
                X.append(hatx + self.h[i] * grad_x)
                # update Z
                hatz = Z[-1] + self.thetaz[i] * (Z[-1] - Z0) # acceleration
                grad_z = L0 + self.beta2[i] * (X[-1] - hatz)
                z_input = (hatz + self.h[i] * grad_z).view(-1, 1, 20, 40)
                x = F.conv2d(z_input, self.conv1_forward, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward, padding=1)
                x = F.conv2d(self.self_active(x_forward, self.soft_thr[i]), self.conv1_backward, padding=1)
                x = F.relu(x)
                x_backward = F.conv2d(x, self.conv2_backward, padding=1)
                Z.append(x_backward.view(-1, 800))
                # update L
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L0) # acceleration
                L.append(hatL + self.h[i] * self.beta1[i]*(X[-1] - Z[-1])) 
                # A^TA
                x = F.conv2d(x_forward, self.conv1_backward, padding=1)
                x = F.relu(x)
                x_sym = F.conv2d(x, self.conv2_backward, padding=1)
                layers_sym.append(x_sym - z_input)
            else:
                # update X
                hatx = X[-1] + self.thetax[i] * (X[-1] - X[-2]) # acceleration
                grad_x = PhiTb - torch.mm(hatx, PhiTPhi) + (self.beta1[i] * (Z[-1] - hatx) - L[-1])
                X.append(hatx + self.h[i] * grad_x)
                # update Z
                hatz = Z[-1] + self.thetaz[i] * (Z[-1] - Z[-2]) # acceleration
                grad_z = L[-1] + self.beta2[i] * (X[-1] - hatz)
                z_input = (hatz + self.h[i] * grad_z).view(-1, 1, 20, 40)
                x = F.conv2d(z_input, self.conv1_forward, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward, padding=1) 
                x = F.conv2d(self.self_active(x_forward, self.soft_thr[i]), self.conv1_backward, padding=1)
                x = F.relu(x)
                x_backward = F.conv2d(x, self.conv2_backward, padding=1)
                Z.append(x_backward.view(-1, 800))
                # update L
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L[-2]) # acceleration
                L.append(hatL + self.h[i] * self.beta1[i] * (X[-1] - Z[-1])) 
                # A^TA
                x = F.conv2d(x_forward, self.conv1_backward, padding=1)
                x = F.relu(x)
                x_sym = F.conv2d(x, self.conv2_backward, padding=1)
                layers_sym.append(x_sym - z_input)
       
        X_final, Z_final, L_final = X, Z, L

        return [Z_final, layers_sym, Wloss]

class A_TLADMMNet(torch.nn.Module):
    def __init__(self, LayerNo, Phi):
        super(A_TLADMMNet, self).__init__()
        self.LayerNo = LayerNo
        self.W = nn.Parameter(Phi)
        self.conv_size = 32
        self.beta1 = nn.Parameter(torch.ones(self.LayerNo))
        self.beta2 = nn.Parameter(torch.ones(self.LayerNo)) 
        self.h = nn.Parameter(torch.ones(self.LayerNo))
        self.h1 = nn.Parameter(torch.ones(self.LayerNo))
        self.h2 = nn.Parameter(torch.ones(self.LayerNo))
        self.soft_thr_pre = nn.Parameter(0.01*torch.ones(self.LayerNo))
        self.soft_thr = nn.Parameter(0.01*torch.ones(self.LayerNo))
        self.thetax = nn.Parameter(0.1*torch.ones(self.LayerNo))
        self.thetaz = nn.Parameter(0.1*torch.ones(self.LayerNo))
        self.thetaL = nn.Parameter(0.1*torch.ones(self.LayerNo))
        
        self.conv1_forward_pre = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward_pre = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward_pre = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward_pre = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3))) 
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3))) 
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3))) 
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3))) 


    def self_active(self, x, thershold):
        s = torch.abs(thershold)
        return torch.mul(torch.sign(x), F.relu(torch.abs(x) - s))

    def measure_x(self, x):
        y = torch.einsum("ma,ba->bm", self.W, x)
        n = 1e-4*torch.randn_like(y)
        y = y+n
        return y

    def forward(self, y, x_original):
        y = y.view(y.size(0), -1)
        PhiTPhi = torch.mm(self.W.t(), self.W)
        PhiTb = torch.mm(y, self.W)

        min_x = torch.min(x_original)
        max_x = torch.max(x_original)
        
        X0 = torch.zeros((y.size(0), 800)).cuda()
        Z0 = torch.zeros((y.size(0), 800)).cuda()
        L0 = torch.zeros((y.size(0), 800)).cuda()
        
        X = list()
        Z = list()
        L = list()
        layers_sym = list()
        Wloss = torch.mm(self.W, self.W.t()) - torch.eye(self.W.shape[0], self.W.shape[0]).cuda()
        
        # 0.5*\|y-phix\|_2^2 + lambda*\|Ax\|_1 -----> 0.5*\|y-phix\|_2^2 + lambda*\|Az\|_1   z = x
        for i in range(self.LayerNo):
            if i == 0:
                # update X
                grad_x = PhiTb - torch.mm(X0, PhiTPhi) + (self.beta1[i] * (Z0 - X0) - L0)
                x_pre = X0 + self.h[i] * grad_x
                grad_x_pre = PhiTb - torch.mm(x_pre, PhiTPhi) + (self.beta1[i] * (Z0 - x_pre) - L0)
                X.append(X0 + self.h1[i] * (grad_x + grad_x_pre))
                # update Z
                grad_z = L0 + self.beta2[i] * (X[-1] - Z0)
                z_pre = (Z0 + self.h[i] * grad_z)

                x = F.conv2d(z_pre.view(-1, 1, 20, 40), self.conv1_forward_pre, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward_pre, padding=1)  
                x = F.conv2d(self.self_active(x_forward, self.soft_thr_pre[i]), self.conv1_backward_pre, padding=1)
                x = F.relu(x)
                z_pre = F.conv2d(x, self.conv2_backward_pre, padding=1)

                grad_z_pre = L0 + self.beta2[i] * (X[-1] - z_pre.view(-1, 800))
                z_input = (Z0 + self.h2[i] * (grad_z + grad_z_pre)).view(-1, 1, 20, 40)
                x = F.conv2d(z_input, self.conv1_forward, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward, padding=1) 
                x = F.conv2d(self.self_active(x_forward, self.soft_thr[i]), self.conv1_backward, padding=1)
                x = F.relu(x)
                x_backward = F.conv2d(x, self.conv2_backward, padding=1)
                Z.append(x_backward.view(-1, 800))
                # update L
                L.append(L0 + self.h[i] * self.beta1[i]*(X[-1] - Z[-1])) 
                # A^TA
                x = F.conv2d(x_forward, self.conv1_backward, padding=1)
                x = F.relu(x)
                x_sym = F.conv2d(x, self.conv2_backward, padding=1)
                layers_sym.append(x_sym - z_input)

            elif i == 1:
                # update X
                hatx = X[-1] + self.thetax[i] * (X[-1] - X0) # acceleration
                grad_x = PhiTb - torch.mm(hatx, PhiTPhi) + (self.beta1[i] * (Z0 - hatx) - L0)
                x_pre = hatx + self.h[i] * grad_x
                grad_x_pre = PhiTb - torch.mm(hatx, PhiTPhi) + (self.beta1[i] * (Z0 - x_pre) - L0)
                X.append(hatx + self.h1[i] * (grad_x + grad_x_pre))
                # 更新 Z
                hatz = Z[-1] + self.thetaz[i] * (Z[-1] - Z0) # acceleration
                grad_z = L0 + self.beta2[i] * (X[-1] - hatz)
                z_pre = (hatz + self.h[i] * grad_z)
                x = F.conv2d(z_pre.view(-1, 1, 20, 40), self.conv1_forward_pre, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward_pre, padding=1)
                x = F.conv2d(self.self_active(x_forward, self.soft_thr_pre[i]), self.conv1_backward_pre, padding=1)
                x = F.relu(x)
                z_pre = F.conv2d(x, self.conv2_backward_pre, padding=1)
                grad_z_pre = L0 + self.beta2[i] * (X[-1] - z_pre.view(-1, 800))
                z_input = (hatz + self.h2[i] * (grad_z + grad_z_pre)).view(-1, 1, 20, 40)
                x = F.conv2d(z_input, self.conv1_forward, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward, padding=1) 
                x = F.conv2d(self.self_active(x_forward, self.soft_thr[i]), self.conv1_backward, padding=1)
                x = F.relu(x)
                x_backward = F.conv2d(x, self.conv2_backward, padding=1)
                Z.append(x_backward.view(-1, 800))
                # update L
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L0) # acceleration
                L.append(hatL + self.h[i] * self.beta1[i]*(X[-1] - Z[-1])) 

                # A^TA
                x = F.conv2d(x_forward, self.conv1_backward, padding=1)
                x = F.relu(x)
                x_sym = F.conv2d(x, self.conv2_backward, padding=1)
                layers_sym.append(x_sym - z_input)

            else:
                # update X
                hatx = X[-1] + self.thetax[i] * (X[-1] - X[-2]) # acceleration
                grad_x = PhiTb - torch.mm(hatx, PhiTPhi) + (self.beta1[i] * (Z[-1] - hatx) - L[-1])
                x_pre = hatx + self.h[i] * grad_x
                grad_x_pre = PhiTb - torch.mm(hatx, PhiTPhi) + (self.beta1[i] * (Z[-1] - x_pre) - L[-1])
                X.append(hatx + self.h1[i] * (grad_x + grad_x_pre))
                # update Z
                hatz = Z[-1] + self.thetaz[i] * (Z[-1] - Z[-2]) # acceleration
                grad_z = L[-1] + self.beta2[i] * (X[-1] - hatz)
                z_pre = (hatz + self.h[i] * grad_z)

                x = F.conv2d(z_pre.view(-1, 1, 20, 40), self.conv1_forward_pre, padding=1)
                x = F.relu(x)
                x_forward = F.conv2d(x, self.conv2_forward_pre, padding=1) 
                x = F.conv2d(self.self_active(x_forward, self.soft_thr_pre[i]), self.conv1_backward_pre, padding=1)
                x = F.relu(x)
                z_pre = F.conv2d(x, self.conv2_backward_pre, padding=1)

                grad_z_pre = L[-1] + self.beta2[i] * (X[-1] - z_pre.view(-1, 800))
                z_input = (hatz + self.h2[i] * (grad_z + grad_z_pre)).view(-1, 1, 20, 40)
                x = F.conv2d(z_input, self.conv1_forward_pre, padding=1)
                x = F.relu(x) 
                x_forward = F.conv2d(x, self.conv2_forward_pre, padding=1) 
                x = F.conv2d(self.self_active(x_forward, self.soft_thr[i]), self.conv1_backward_pre, padding=1)
                x = F.relu(x)
                x_backward = F.conv2d(x, self.conv2_backward_pre, padding=1)
                Z.append(x_backward.view(-1, 800))
                # update L
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L[-2]) # acceleration
                L.append(hatL + self.h[i] * self.beta1[i] * (X[-1] - Z[-1])) 

                # A^TA
                x = F.conv2d(x_forward, self.conv1_backward, padding=1)
                x = F.relu(x)
                x_sym = F.conv2d(x, self.conv2_backward, padding=1)
                layers_sym.append(x_sym - z_input)
        
        X_final, Z_final, L_final = X, Z, L

        return [Z_final, layers_sym, Wloss]

class MeasurementsDataset(Dataset):
    def __init__(self, measurements_data_folder, split="train"):
        db_path = os.path.join(measurements_data_folder, f"{split}.lmdb")
        self.env = lmdb.open(
            db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        y, x = pickle.loads(byteflow)
        return y, x
train = MeasurementsDataset(data_path, split = 'train')
test = MeasurementsDataset(data_path, split = 'test')

rand_loader = DataLoader(dataset = train, batch_size = batch_size, num_workers = 4, shuffle = True)
rand_val = DataLoader(dataset = test, batch_size = batch_size, num_workers = 2, shuffle = True)
if args.dataset == "speechcommands":
    evaluation = MYSPEECHCOMMANDS(root="./data", subset="testing", sample_rate=args.sample_rate, max_length=MAX_LENGTH,)
    evaluation_loader = DataLoader(evaluation, num_workers=1, batch_size=batch_size, shuffle = False,)
else:
    evaluation = Timit(
            data_path="./data/timit",
            split="test",
            sample_rate=args.sample_rate,
            max_length=MAX_LENGTH,
        )
    evaluation_loader = DataLoader(
        evaluation,
        num_workers=1,
        batch_size=batch_size,
        shuffle=False,
    )

if args.alg == "A-ELADMM-Net":
    model_dir = "./%s/A_ELADMM_Audio_layer_%d_lr_%.4f" % (args.model_dir, layers, learning_rate)
    model = A_ELADMMNet(layers, Phi)
if args.alg == "A-TLADMM-Net":
    model_dir = "./%s/A_TLADMM_Audio_layer_%d_lr_%.4f" % (args.model_dir, layers, learning_rate)
    model = A_TLADMMNet(layers, Phi)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

print("./%s/%s_%.2f_%s_net_params_%d.pkl" % (model_dir, args.dataset, Phi.shape[0]/AMBIENT_DIM, args.alg, start_epoch))
if start_epoch > 0:
    model.load_state_dict(torch.load("./%s/%s_%.2f_%s_net_params_%d.pkl" % (model_dir, args.dataset, Phi.shape[0]/AMBIENT_DIM, args.alg, start_epoch)))

for epoch_i in range(start_epoch+1, epochs):
    # train You can annotate this training code and directly test the trained model in the folder "model/".
    avg_train_mse = 0
    for batch in rand_loader:
        y, x_original = batch
        y = y.cuda()
        x_original = x_original.cuda()
        x_hat, constraint, Wloss = model(y, x_original)
        criterion = nn.MSELoss()
        Discrepancy_Loss = criterion(x_hat[0], x_original.view(x_original.size(0), -1))
        for i in range(layers-1):
            Discrepancy_Loss += (i+1)*criterion(x_hat[i+1], x_original.view(x_original.size(0), -1))
        loss_constraint = torch.mean(torch.pow(constraint[0], 2))
        for k in range(layers-1):
            loss_constraint += torch.mean(torch.pow(constraint[k+1], 2))
        Total_Loss = Discrepancy_Loss + 0.01 * loss_constraint + torch.mean(torch.pow(Wloss, 2))
        optimizer.zero_grad()
        Total_Loss.backward()
        optimizer.step()
        avg_train_mse += Discrepancy_Loss
    output_data = "[%02d/%02d] Total Loss: %.6f, avg_train_mse: %.6f,  Constraint Loss: %.6f,  WLoss: %.6f\n" % (epoch_i+1, epochs, Total_Loss, Discrepancy_Loss, loss_constraint, torch.mean(torch.pow(Wloss, 2)))
    print(output_data)
    avg_train_mse /= len(rand_loader)
    
    if (epoch_i+1) % 10 == 0:
        torch.save(model.state_dict(), "./%s/%s_%.2f_%s_net_params_%d.pkl" % (model_dir, args.dataset, Phi.shape[0]/AMBIENT_DIM, args.alg, (epoch_i+1)))
    # test
    if (epoch_i+1) % 10 == 0:
        model.load_state_dict(torch.load("%s/%s_%.2f_%s_net_params_%d.pkl" % (model_dir, args.dataset, Phi.shape[0]/AMBIENT_DIM, args.alg, (epoch_i+1))))
        avg_val_mse = 0
        for batch in rand_val:
            y, x_original = batch
            y = y.cuda()
            x_original = x_original.cuda()
            x_hat, constraint, Wloss = model(y, x_original)
            criterion = nn.MSELoss()
            mse = criterion(x_hat[-1], x_original.view(x_original.size(0), -1))
            avg_val_mse += mse.item()
        avg_val_mse /= len(rand_val)
        print("[%02d/%02d] test_mse: %.8f" % (epoch_i+1, epochs, avg_val_mse))
        save_examples(model, evaluation_loader, (epoch_i+1), algo=args.alg, device="cpu")




