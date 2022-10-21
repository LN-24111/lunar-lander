import argparse
import pickle

def handle(number):
    def register(func):
        func_registry[number] = func
        return func

    return register

def run(mode):
    if mode not in func_registry:
        raise ValueError(f"unknown question {mode}")
    return func_registry[mode]()

def main():
    """
        Borrowing this~
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, choices=func_registry.keys())
    args = parser.parse_args()
    return run(args.mode)

import jax
from jax import random

from models import *
from utils import load, store, get_params
from simulator import visualize, simulate, evaluate_fitness


# @handle("train_pg")
# def train_pg():
#     agent = PG()

#     reward_data = []
#     eplen_data = []
#     root_name = input("name?")

#     for i in range(LIFETIME_LIMIT):
#         observations, actions, rewards, total_reward, n = simulate(agent)
#         agent.train(observations, actions, rewards)

#         reward_data.append(total_reward)
#         eplen_data.append(n)

#         if i%100 == 0:
#             print (i//100, total_reward)
#             # if root_name != "":
#             #     with open(f"pg_{root_name}_{i//100}_data", "wb") as f:
#             #         pickle.dump((reward_data, eplen_data), f)

#             #     with open(f"pg_{root_name}_{i//100}_model", "wb") as f:
#             #         pickle.dump((agent.params, agent.vparams), f)

#     if root_name != "":
#         with open(f"pg_{root_name}_data.pkl", "wb") as f:
#             pickle.dump((reward_data, eplen_data), f)

#         with open(f"pg_{root_name}_model.pkl", "wb") as f:
#             pickle.dump((agent.params, agent.vparams), f)

#     print(visualize(agent, 1))


@handle("load_pg")
def load_pg():
    root_name = input("name? ")
    agent = PG()
    with open(f"pg_{root_name}_model.pkl", "rb") as f:
        agent.params, agent.vparams, lr, vlr, gamma = pickle.load(f)

    print(visualize(agent, 5))
@handle("train_ga")
def train_ga():
    pass

@handle("load_ga")
def load_ga():
    root_name = input("name? ")
    
    with open(f"ga_{root_name}_model.pkl", "rb") as f:
        params = pickle.load(f)

    print(visualize(GAWrap(params), 5))

if __name__ == "__main__":
    main()
    
 
    
