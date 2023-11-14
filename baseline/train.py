# !/usr/bin/python3

# importing necessary libraries
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time
import tensorboardX

# to get the current experience name
exp = os.path.abspath('.').split('/')[-1]

# Tensorboard writer to log the training process
writer = tensorboardX.SummaryWriter(os.path.join('train_log', exp))
# symbolic link to the training log directory
os.system('ln -sf ../train_log/{} ./log'.format(exp))
# creating directory to store trained model
os.system('mkdir ./model')

def train(agent, env, evaluate):
    """Trains the agent.
    Args:
        agent: The agent to train.
        env: The environment in which to train the agent.
        evaluate: A function that evaluates the agent's performance.
    Returns:
        None.
    """
    # Getting the training paramenters
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()

    # initializing the episode counter
    step = episode = episode_steps = 0
    # total reward
    tot_reward = 0.
    # Jnitializing observation
    observation = None
    noise_factor = args.noise_factor

    # Starting the training loop
    while step <= train_times:
        step += 1
        episode_steps += 1
        # reset environment and agent if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)   
        # selecting action 
        action = agent.select_action(observation, noise_factor=noise_factor)

        # Taking step in environment
        observation, reward, done, _ = env.step(action)

        # Add experience to agent's memory
        agent.observe(reward, observation, done, step)

        # if episode over, evaluate agent and save the model
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    if debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.

            # Updating agent's policy and value networks
            if step > args.warmup:
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
            if debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')
    # Argument parsers allow you to parse command-line arguments and pass them to your program.

    # hyper-parameter
    # Hyperparameters are parameters that control the training process, such as the learning rate, the discount factor, and the batch size.
    parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')


    # parses the command-line arguments and returns a dictionary containing the parsed arguments
    args = parser.parse_args() 

    # gets output folder for the model
    args.output = get_output_folder(args.output, "Paint")

    # ensures that the program produces the same results every time it is run
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


    from DRL.ddpg import DDPG
    from DRL.multi import fastenv

    # creating environment
    fenv = fastenv(args.max_step, args.env_batch, writer)

    # creating agent
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output)
    
    # creating evaluator
    evaluate = Evaluator(args, writer)
    
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    train(agent, fenv, evaluate)
