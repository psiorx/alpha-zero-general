import os
import sys
import time

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .Connect4NNet import Connect4NNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'minimal_logging': False,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.args = args
        self.nnet = onnet(game, self.args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        first_loss = None
        last_loss = None
        grad_norms = []

        for epoch in range(self.args.epochs):
            if not self.args.minimal_logging:
                print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            use_tqdm = (not self.args.minimal_logging) and tqdm is not None
            loop = tqdm(range(batch_count), desc='Training Net') if use_tqdm else range(batch_count)
            for _ in loop:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                if first_loss is None:
                    first_loss = (l_pi.item(), l_v.item())
                last_loss = (l_pi.item(), l_v.item())

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                if use_tqdm:
                    loop.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                
                # Compute gradient norm
                grad_norm = 0.0
                for p in self.nnet.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                grad_norms.append(grad_norm)
                
                optimizer.step()
        if first_loss is None or last_loss is None:
            return None
        # Return tuple format for backward compatibility, plus extended info
        self.last_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        return first_loss, last_loss

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            if not self.args.minimal_logging:
                print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        elif not self.args.minimal_logging:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
