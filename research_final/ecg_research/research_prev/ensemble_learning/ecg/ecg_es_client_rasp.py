#!/usr/bin/env python
# coding: utf-8

users = 3 # number of clients

import os
import struct
import socket
import pickle
import time

import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import copy

root_path = '../../../models/'

def getFreeDescription():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 1:
            return (line.split()[0:7])


def getFree():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 2:
            return (line.split()[0:7])

# from gpiozero import CPUTemperature


# def printPerformance():
#     cpu = CPUTemperature()
#
#     print("temperature: " + str(cpu.temperature))
#
#     description = getFreeDescription()
#     mem = getFree()
#
#     print(description[0] + " : " + mem[1])
#     print(description[1] + " : " + mem[2])
#     print(description[2] + " : " + mem[3])
#     print(description[3] + " : " + mem[4])
#     print(description[4] + " : " + mem[5])
#     print(description[5] + " : " + mem[6])

# printPerformance()

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(777)
if device =="cuda:0":
    torch.cuda.manual_seed_all(777)

client_order = int(input("client_order(start from 0): "))

num_traindata = 13244 // users

class ECG(Dataset):
    def __init__(self, train=True):
        if train:
            # total: 13244
            with h5py.File(os.path.join(root_path, 'ecg_data', 'train_ecg.hdf5'), 'r') as hdf:
                self.x = hdf['x_train'][num_traindata * client_order : num_traindata * (client_order + 1)]
                self.y = hdf['y_train'][num_traindata * client_order : num_traindata * (client_order + 1)]

        else:
            with h5py.File(os.path.join(root_path, 'ecg_data', 'test_ecg.hdf5'), 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])

batch_size = 32

train_dataset = ECG(train=True)
test_dataset = ECG(train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

x_train, y_train = next(iter(train_loader))
print(x_train.size())
print(y_train.size())

total_batch = len(train_loader)
print(total_batch)

class Ecgclient(nn.Module):
    def __init__(self):
        super(Ecgclient, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x   

ecg_client = Ecgclient().to(device)
print(ecg_client)

epoch = 20  # default
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = Adam(ecg_client.parameters(), lr=lr)

def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

host = input("IP address: ")
port = 10080

start_time = time.time()    # store start time
print("timmer start!")

s = socket.socket()
s.connect((host, port))

msg = recv_msg(s)   # get epoch
epochs = msg['epochs']
users = msg['users']
msg = total_batch
send_msg(s, msg)   # send total_batch of train dataset


for e in range(epochs):
    for u in range(users):
        client_weights = recv_msg(s)
        ecg_client.load_state_dict(client_weights)
        ecg_client.eval()
        for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Epoch '+str(e+1)+ '_' +str(u))):
            x, label = data
            x = x.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = ecg_client(x)
            client_output = output.clone().detach().requires_grad_(True)
            msg = {
                'client_output': client_output,
                'label': label
            }
            send_msg(s, msg)
            client_grad = recv_msg(s)
            output.backward(client_grad)
            optimizer.step()
        msg = copy.deepcopy(ecg_client.state_dict())
        send_msg(s, msg)

end_time = time.time()  #store end time
print("WorkingTime of ",device ,": {} sec".format(end_time - start_time))

