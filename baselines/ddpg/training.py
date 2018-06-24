import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def replay_with_goal(traj, goal, env):
    # start the hindsight trajectory in the last episode that occured in the trajectory.
    for (obs, action, r, new_obs, done) in traj:
        obs_hindsight = env.change_goal(goal, obs)
        action_hindsight = action
        r_hindsight = env.compute_reward(goal, new_obs)
        new_obs_hindsight = env.change_goal(goal, new_obs)
        done_hindsight = env.compute_terminal(goal, new_obs_hindsight)
        yield obs_hindsight, action_hindsight, r_hindsight, new_obs_hindsight, done_hindsight

def replay_final(traj, env):
    episodes = []
    current_episode = []
    for (obs, action, r, new_obs, done) in traj:
        current_episode.append((obs, action, r, new_obs, done))
        if done:
            episodes.append(current_episode)
            current_episode = []
    for episode in episodes:
        obs_final, action_final, r_final, new_obs_final, done_final = episode[-1]
        goal_final = env.obs_to_goal(new_obs_final)
        for (obs, action, r, new_obs, done) in replay_with_goal(episode, goal_final, env):
            yield (obs, action, r, new_obs, done)


def replay_future(traj, env, k=4):
    episodes = []
    current_episode = []
    for (obs, action, r, new_obs, done) in traj:
        current_episode.append((obs, action, r, new_obs, done))
        if done:
            episodes.append(current_episode)
            current_episode = []
    for i in range(k):
        for episode in episodes:
            random_idx = np.random.randint(0, len(episode))
            obs_random, action_random, r_random, new_obs_random, done_random = episode[random_idx]
            goal_random = env.obs_to_goal(new_obs_random)
            for (obs, action, r, new_obs, done) in replay_with_goal(episode[:random_idx+1], goal_random, env):
                yield (obs, action, r, new_obs, done)



def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, save_path=None, restore=False, hindsight_mode=None):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None
    
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        if restore_path is not None:
            logger.info("Restoring from saved model")
            saver.restore(sess, tf.train.latest_checkpoint(save_path))

        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()


        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                transitions = []
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    transitions.append((obs, action, r, new_obs, done))
                    #agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # store regular transitions into replay memory
                for (obs, action, r, new_obs, done) in transitions:
                    agent.store_transition(obs, action, r, new_obs, done)

                if hindsight_mode in ['final', 'future']:
                    for (obs, action, r, new_obs, done) in replay_final(transitions, env.env):
                        agent.store_transition(obs, action, r, new_obs, done)

                if hindsight_mode in ['future']:
                    for (obs, action, r, new_obs, done) in replay_future(transitions, env.env):
                        agent.store_transition(obs, action, r, new_obs, done)

                # store hindsight transitions.
                '''for i in range(3):
                    # sample a random point in the trajectory
                    idx = np.random.randint(0, len(transitions))
                    obs, action, r, new_obs, done = transitions[idx]
                    # create a goal from that point
                    goal = env.env.obs_to_goal(new_obs)
                    for (obs, action, r, new_obs, done) in replay_with_goal(transitions[:idx+1], goal, env.env):
                        agent.store_transition(obs, action, r, new_obs, done)
                obs, action, r, new_obs, done = transitions[-1]

                # store a "final" transition.
                goal = env.env.obs_to_goal(new_obs)
                for (obs, action, r, new_obs, done) in replay_with_goal(transitions, goal, env.env):
                    agent.store_transition(obs, action, r, new_obs, done)'''

                # Train.

                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    # agent.train_planning()
                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)
    
            # Train statistics.
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = mpi_mean(eval_qs)
                combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()

            if t % 1000 == 0 and rank == 0:
	            logger.info('saving model to: {}'.format(save_path))
	            saver.save(sess, save_path, global_step=epoch, write_meta_graph=False)
	            logger.info('done saving model!')
            
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

