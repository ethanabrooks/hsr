#!/usr/bin/env python
import argparse
from baselines import bench, logger
from environment.arm2pos import Arm2Pos
from environment.navigate import NavigateEnv
from environment.pick_and_place import PickAndPlaceEnv
from toy_environment import continuous_gridworld, continuous_gridworld2

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        if env_id == 'toy':
            #env = continuous_gridworld.ContinuousGridworld('', max_steps=1000,
            #                                           obstacle_mode=continuous_gridworld.NO_OBJECTS)
            env = continuous_gridworld2.ContinuousGridworld2()
        elif env_id == 'navigate':
            env = NavigateEnv(use_camera=False, continuous_actions=True, neg_reward=True, max_steps=500)
        elif env_id == 'arm2pos':
            env = Arm2Pos(use_camera=False, continuous=False, max_steps=500)
        else:
            env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('--tb-dir', default=None)
    args = parser.parse_args()
    if args.tb_dir is not None:
        logger.configure(dir=args.tb_dir, format_strs=['stdout', 'tensorboard'])
    else:
        logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)



if __name__ == '__main__':
    main()

