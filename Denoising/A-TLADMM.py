############################################### Public verison ####################################

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import scipy.misc 
import imageio
import time

class ATLADMM(nn.Module):
    def __init__(self, m, n, d, batch_size, A, x0, y0, L0, layers):
        super(ATLADMM, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        self.batch_size = batch_size
        self.A = A.cuda()
        self.x0 = x0.cuda()
        self.y0 = y0.cuda()
        self.L0 = L0.cuda()
        self.layers = layers
        self.W = nn.ParameterList()
        self.beta1 = nn.Parameter(0.01*torch.ones(layers))
        self.theta = nn.Parameter(0.1*torch.ones(layers))
        self.eta = nn.Parameter(0.05*torch.ones(layers))
        self.active_para = nn.Parameter(0.01*torch.ones(layers))
        self.active_para1 = nn.Parameter(0.01*torch.ones(layers))
        self.h0 = nn.Parameter(0.05*torch.ones(layers))
        for k in range(self.layers):
            self.W.append(nn.Parameter(self.A + 0.5*torch.randn_like(self.A)))

    def self_active(self, x, thershold):
        s = torch.abs(thershold)
        return torch.mul(torch.sign(x), F.relu(torch.abs(x) - s))

    def forward(self, b):
        Var = list()
        x = list()
        y = list()
        L = list()
        xhat = list()
        yhat = list()
        for k in range(self.layers):
            if k == 0 :
                Var.append(self.L0 + self.beta1[k] * (self.A.mm(self.x0) + self.y0 - b))
                xhat.append(self.self_active(self.x0 - self.h0[k] / self.theta[k] * self.W[k].t().mm(Var[-1]), self.active_para[k]))
                tempVarx = self.L0 + self.beta1[k] * (self.A.mm(xhat[-1]) + self.y0 - b)
                x.append(self.self_active(self.x0 - 0.5 * self.h0[k] / self.theta[k] * (self.W[k].t().mm(Var[-1] + tempVarx)), self.active_para[k]))
                yhat.append(self.self_active(self.y0 - self.h0[k] / self.eta[k] * (self.y0 - b + self.A.mm(x[-1]) + (self.L0)/self.beta1[k]), self.active_para[k]))
                y.append(self.self_active(self.y0 - self.h0[k] / self.eta[k] * (0.5 * self.y0 - b + self.A.mm(x[-1]) + (self.L0)/self.beta1[k] + 0.5 * yhat[-1]), self.active_para1[k]))   
                L.append(self.L0 + self.h0[k] * self.beta1[k].mul(self.A.mm(x[-1]) + y[-1] - b))

            elif k == 1 :
                hat_x = self.self_active(x[-1] + self.h0[k] / self.theta[k] * (x[-1] - self.x0), self.active_para[k]) #  1/(1 + self.h0[k]*self.theta[k])
                Var.append(self.L0 + self.beta1[k] * (self.A.mm(hat_x) + self.y0 - b))
                xhat.append(self.self_active(hat_x - self.h0[k] / self.theta[k] * self.W[k].t().mm(Var[-1]), self.active_para[k]))
                tempVarx = self.L0 + self.beta1[k] * (self.A.mm(xhat[-1]) + self.y0 - b)
                x.append(self.self_active(hat_x - 0.5 * self.h0[k] / self.theta[k] * (self.W[k].t().mm(Var[-1] + tempVarx)), self.active_para[k]))
                hat_y = self.self_active(y[-1] + self.h0[k] / self.eta[k] * (y[-1] - self.y0), self.active_para1[k]) # 1/(1 + self.h0[k]*self.eta[k])
                yhat.append(self.self_active(hat_y - self.h0[k] / self.eta[k] * (hat_y - b + self.A.mm(x[-1]) + (self.L0)/self.beta1[k]), self.active_para[k]))
                y.append(self.self_active(hat_y - self.h0[k] / self.eta[k] * (0.5 * hat_y - b + self.A.mm(x[-1]) + (self.L0)/self.beta1[k] + 0.5 * yhat[-1]), self.active_para1[k]))   
                hat_L = L[-1] + self.beta1[k]/(self.beta1[k] + self.h0[k])*(L[-1] - self.L0)
                L.append(hat_L + self.h0[k] * self.beta1[k] * (self.A.mm(x[-1]) + y[-1] - b))

            else :
                hat_x = self.self_active(x[-1] + self.h0[k] / self.theta[k] * (x[-1] - x[-2]), self.active_para[k])
                Var.append(L[-1] + self.beta1[k] * (self.A.mm(hat_x) + y[-1] - b))
                xhat.append(self.self_active(hat_x - self.h0[k] / self.theta[k] * self.W[k].t().mm(Var[-1]), self.active_para[k]))
                tempVarx = L[-1] + self.beta1[k] * (self.A.mm(xhat[-1]) + y[-1] - b)
                x.append(self.self_active(hat_x - 0.5 * self.h0[k] / self.theta[k] * (self.W[k].t().mm(Var[-1] + tempVarx)), self.active_para[k]))
                hat_y = self.self_active(y[-1] + self.h0[k] / self.eta[k] * (y[-1] - y[-2]), self.active_para1[k])
                yhat.append(self.self_active(hat_y - self.h0[k] / self.eta[k] * (hat_y - b + self.A.mm(x[-1]) +(L[-1])/self.beta1[k]), self.active_para[k]))
                y.append(self.self_active(hat_y - self.h0[k] / self.eta[k] * (0.5 * hat_y - b + self.A.mm(x[-1]) + (L[-1])/self.beta1[k] + 0.5 * yhat[-1]), self.active_para1[k]))
                hat_L = L[-1] + self.beta1[k]/(self.beta1[k] + self.h0[k])*(L[-1] - L[-2])
                L.append(hat_L + self.h0[k] * self.beta1[k] * (self.A.mm(x[-1]) + y[-1] - b))
        return x, y, L

    def name(self):
        return "A-TLADMM"

def trans2image(img):
	img = img.cpu().data.numpy()
	new_img = np.zeros([512, 512])
	count = 0
	for ii in range(0, 512, 16):
		for jj in range(0, 512, 16):
			new_img[ii:ii+16,jj:jj+16] = np.transpose(np.reshape(img[:, count],[16,16]))
			count = count+1
	return new_img

def l2_normalize(inputs):
    [batch_size, dim] = inputs.shape
    inputs2 = torch.mul(inputs, inputs)
    norm2 = torch.sum(inputs2, 1)
    root_inv = torch.rsqrt(norm2) 
    tmp_var1 = root_inv.expand(dim,batch_size)
    tmp_var2 = torch.t(tmp_var1)
    nml_inputs = torch.mul(inputs, tmp_var2)
    return nml_inputs

def l2_col_normalize(inputs):
    [dim1, dim2] = inputs.shape
    inputs2 = np.multiply(inputs, inputs)
    norm2 = np.sum(inputs2, 0)
    root = np.sqrt(norm2)
    root_inv = 1/root
    tmp_var1 = np.tile(root_inv,dim1)
    tmp_var2 = tmp_var1.reshape(dim1, dim2)
    nml_inputs = np.multiply(inputs, tmp_var2)
    return nml_inputs

def calc_PSNR(x1, x2):
	x1 = x1 * 255.0
	x2 = x2 * 255.0
	mse = F.mse_loss(x1, x2)
	psnr = -10 * torch.log10(mse) + torch.tensor(48.131)
	return psnr

np.random.seed(1126)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
m, d, n = 256, 512, 10000 
n_test = 1024
batch_size = 25
layers = 15
alpha = 0.1
num_epoch = 20

use_cuda = torch.cuda.is_available()
print('==>>> batch size: {}'.format(batch_size))
print('==>>> total trainning batch number: {}'.format(n//batch_size))
print('==>>> total testing batch number: {}'.format(n_test//batch_size))

img_data = sio.loadmat('washsat_pepper_01.mat')
A_ori = img_data['D'] 
A_ori = A_ori.astype(np.float32)

b = img_data['train_x'].astype(np.float32)
b = b.T

b_ts = img_data['test_x'].astype(np.float32)
b_ts = b_ts.T

b_gt = img_data['gt_x'].astype(np.float32)
b_gt = b_gt.T

x0 = torch.zeros(d, batch_size, dtype=torch.float32)
y0 = torch.zeros(m, batch_size, dtype=torch.float32)
L0 = torch.zeros(m, batch_size, dtype=torch.float32)
A_tensor = torch.from_numpy(A_ori) 

model = ATLADMM(m=m, n=n, d=d, batch_size=batch_size, A=A_tensor, x0=x0, y0=y0, L0=L0, layers=layers)

paras = list(model.parameters())
A_tensor = A_tensor.cuda()
if use_cuda:
   model = model.cuda()

criterion = nn.MSELoss()
index_loc = np.arange(10000)
ts_index_loc = np.arange(1000)
psnr_value = 0
best_pic = torch.zeros(256, 1024)

for epoch in range(num_epoch):
    print('---------------------------training---------------------------')
    learning_rate =  0.0002
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    np.random.shuffle(index_loc) 
    for i in range(n//batch_size):
        optimizer.zero_grad()
        address = index_loc[np.arange(i*batch_size,(i+1)*batch_size)]
        input_bs = b[:, address]
        input_bs = torch.from_numpy(input_bs) 
        input_bs_var = torch.autograd.Variable(input_bs.cuda())
        [x, y, L] = model(input_bs_var) 

        loss = list()
        total_loss = 0
        for k in range(layers):
            loss.append((alpha * torch.mean(torch.abs(x[k])) + torch.mean(torch.abs(torch.mm(A_tensor, x[k]) - input_bs_var))))
            total_loss = total_loss + (k+1) * loss[-1] / np.sum(range(layers))

        total_loss.backward() 
        optimizer.step()
        if (i) % 100 == 0 and i != 0:
            print('==>> epoch: {} [{}/{}]'.format(epoch+1, i, n//batch_size))
            for k in range(layers):
                print('loss{}:{:.3f}'.format(k + 1, loss[k]), end=' ')
            print(" ")
    # torch.save(model.state_dict(), model.name()) 
    print('---------------------------testing---------------------------')
    mse_value = torch.zeros(layers)
    Time = []
    with torch.no_grad():
        for j in range(n_test//batch_size):
            input_bs = b_ts[:, j*batch_size:(j+1)*batch_size]
            input_bs = torch.from_numpy(input_bs)
            input_bs_var = torch.autograd.Variable(input_bs.cuda())
            time_start=time.time()
            [x, y, L] = model(input_bs_var)
            time_end=time.time()
            Time.append(time_end-time_start)
            
            input_gt = b_gt[:, j*batch_size:(j+1)*batch_size]
            input_gt = torch.from_numpy(input_gt)
            input_gt_var = torch.autograd.Variable(input_gt.cuda())

            for jj in range(layers):
                mse_value[jj] = mse_value[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, x[jj]))
    print('Time cost on testing dataset: %.4fs' % np.sum(Time))

    mse_value = mse_value / (n_test//batch_size)
    psnr = -10 * torch.log10(mse_value) + torch.tensor(48.131)
    with torch.no_grad():
        for jj in range(layers):
            if(psnr_value < psnr[jj]):
                psnr_value = psnr[jj]
                for jjj in range(n_test//batch_size):
                    input_bs = b_ts[:, jjj*batch_size:(jjj+1)*batch_size]
                    input_bs = torch.from_numpy(input_bs)
                    input_bs_var = torch.autograd.Variable(input_bs.cuda())
                    [x, y, L] = model(input_bs_var)
                    best_pic[:, jjj*batch_size:(jjj+1)*batch_size] = 255* torch.mm(A_tensor, x[jj])

    print('==>> epoch: {}'.format(epoch))
    for k in range(layers):
                print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
    print(" ")
    print('******Best PSNR:{:.3f}'.format(psnr_value))

        



