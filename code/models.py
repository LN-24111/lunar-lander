from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, random

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np
import optax                           # Optimizers

import pickle

class LunarAgent:
    def policy(self, observation):
        raise NotImplementedError

class Dumbass(LunarAgent):
    def policy(self, ignore):
        return 1

class PG(LunarAgent):
    def __init__(self, learning_rate=0.002, vlearning_rate=0.002, gamma=0.995):
        """
            @epsilon:       Likelihood of selecting randomly
            @epsilon_decay:  
            @learning_rate: 
            @gamma:         Future discount
        """
        actor_model, value_model = PGA_DNN(), PGV_DNN()
        actor_param = init_model(actor_model)
        value_param = init_model(value_model)

        self.model = actor_model,
        self.vmodel = value_model
        self.params = actor_param
        self.vparams = value_param

        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(actor_param)

        self.voptimizer = optax.adam(learning_rate)
        self.vopt_state = self.optimizer.init(value_param)

        self.learning_rate = learning_rate
        self.vlearning_rate = vlearning_rate
        self.gamma = gamma

        @jit
        def val_func(params, observations):
            return jnp.ravel(self.vmodel.apply(params, observations))
        self.val_func = val_func

        @jit
        def val_loss(params, observations, values):
            predict = val_func(params, observations)
            loss = jnp.sum(
                    optax.l2_loss(predict, values)
                )
            return loss, predict
        self.val_loss = val_loss
        self.val_grad = jax.value_and_grad(val_loss, has_aux=True)

        @jit
        def policy_func(params, observations):
            return  actor_model.apply(params, observations)
        self.policy_func = policy_func

        @jit
        def policy_loss(params, observations, actions, advantage, gamma_arr):
            probs = policy_func(params, observations)
            sel_probs = jnp.take_along_axis(probs, jnp.expand_dims(actions, axis=1), axis=1)
            sel_probs = jnp.ravel(sel_probs)

            loss_vec = jnp.log(sel_probs) * advantage * gamma_arr
            loss = -jnp.sum(loss_vec)
            return loss
        self.policy_grad = grad(policy_loss)

        @jit
        def train(observations, actions, rewards):
            n = len(observations)
            observations = jnp.array(observations)
            actions = jnp.array(actions)

            # Calculate the Gis
            g = []
            gi = 0
            for i in range(n-1,-1,-1):
                gi = gi * self.gamma + rewards[i]
                g.append(gi)
            g = jnp.array(g)

            # Get the value net prediction and gradient
            (loss, predict), val_grad = self.val_grad(self.vparams, observations, g)
            advantage = g - predict

            vupdates, self.vopt_state = self.voptimizer.update(val_grad, self.vopt_state)
            self.vparams = optax.apply_updates(self.vparams, vupdates)


            gamma_arr = jnp.array([self.gamma ** i for i in range(n)])
            actor_grad = self.policy_grad(self.params, observations, actions, advantage, gamma_arr)

            updates, self.opt_state = self.optimizer.update(actor_grad, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)
            
            # Manual sum
            # total_grad = self.prob_func_grad(self.params, observations[0], actions[0])
            # for i in range(1,n):
            #     actor_grad = self.prob_func_grad(self.params, observations[i], actions[i])
            #     total_grad = jax.tree_util.tree_map(
            #         lambda p, a: p + advantage[i] * self.gamma**i * a, total_grad, actor_grad
            #     )
            # actor_grad = jax.tree_util.tree_map(
            #     lambda p: -p / n, total_grad
            # )
        self.train = train
            

    def policy(self, observation):
        probs = self.policy_func(self.params, observation)
        probs = np.array(probs)
        probs[-1] += 1 - np.sum(probs)
        # print(self.val_func(self.vparams, observation))
        # print(probs)
        choice = np.random.choice(4, 1, p=probs)
        # print(choice)
        return int(choice)

    # @partial(jit, static_argnums=(0,))




class GA(LunarAgent):
    def __init__(self):
        model = GA_DNN()

        self.model = model
        self.params = init_model(model)

    def policy(self, observation):
        return self.model.apply(self.params, observation)

    def offspring(self):
        pass

class GAWrap:
    """
        Wrapper to keep things consistent but we only care about the params
    """
    def __init__(self, params):
        self.params = params
        @jit
        def policy(params, observation):
            return np.argmax( GA_DNN().apply(params, observation) )
        self.policy_func = policy
    
    def policy(self, observation):
        jm= self.policy_func(self.params, observation)
        return int(np.array(jm))

def init_model(model):
    """
        Semi-hardcoded init model
    """
    key1, key2 = random.split(random.PRNGKey(0))
    dummy_input = random.normal(key1, (8,))
    params = model.init(key2, dummy_input)
    
    return params

class PGA_DNN(nn.Module):
    """
        PG Actor Approximator Function
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        x = nn.softmax(x)

        return x

class GA_DNN(nn.Module):
    """
        GA Approximator Function
        Same as PG but without softmax
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)

        return x

class PGV_DNN(nn.Module):
    """
        PG Value Approximator Function
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)

        return x