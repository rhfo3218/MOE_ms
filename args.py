import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--expert_input_size', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=2)
    parser.add_argument('--gate_hidden_size', type=int, default=6)
    parser.add_argument('--expert_output_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--only_for_gate_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lamb', type=int, default=0)
    args = parser.parse_args()
    return args



