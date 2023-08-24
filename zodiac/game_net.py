import copy
import logging

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

log = logging.getLogger(__name__)

args = {
    "num_channels": 128,
    "batch_size": 128,
    "epochs": 10,
}


class ZodNet(nn.Module):
    def __init__(self, game, kernel_size=3, stride=1, padding=1):
        super(ZodNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        input_channels_state_dim = game.getStateDimension()

        self.conv1 = nn.Conv2d(input_channels_state_dim, args["num_channels"], kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(args["num_channels"], args["num_channels"], kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(args["num_channels"], args["num_channels"], kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(args["num_channels"], args["num_channels"], kernel_size, stride=stride, padding=padding)

        conv_output_row = int((self.board_x - kernel_size + 2 * padding) / stride) + 1
        conv_output_col = int((self.board_y - kernel_size + 2 * padding) / stride) + 1
        self.fc1 = nn.Linear(args["num_channels"] * conv_output_row * conv_output_col, 512)
        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512, self.action_size * np.max(game.getInitUnitNumber()))
        self.fc4 = nn.Linear(512, 1)

        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='tanh')

    def forward(self, board_state):
        out = torch.relu(self.conv1(board_state))
        out = torch.relu(self.conv2(out))
        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))

        out = out.reshape(out.size()[0], -1)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))

        pi = torch.softmax(self.fc3(out), dim=1)
        v = torch.tanh(self.fc4(out))

        return pi, v


class Net:
    def __init__(self, game):
        self.game = game
        self.nnet = ZodNet(game).cuda()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def train(self, examples):
        examples = copy.deepcopy(examples)
        random.shuffle(examples)
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args["epochs"]):
            self.nnet.train()

            batch_count = np.ceil(len(examples) / args["batch_size"]).astype(int)
            for count in tqdm(range(batch_count), desc="Epoch " + str(epoch + 1) + "/" + str(args["epochs"])):
                boards, pis, vs = list(zip(*[examples[i] for i in range(count * args["batch_size"], min((count + 1) * args["batch_size"], len(examples)))]))
                boards = [np.transpose(b, (2, 0, 1)) for b in boards]
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis).astype(np.float64))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        board = board.contiguous().cuda()
        board = board.unsqueeze(0)
        board = board.permute(0, 3, 1, 2)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return pi.data.cpu().numpy(), v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            log.error("Model Not Found.")
            return False
        checkpoint = torch.load(filepath, map_location=None if torch.cuda.is_available() else 'cpu')
        self.nnet.load_state_dict(checkpoint['state_dict'])
        log.info("Model Loaded.")
        return True
