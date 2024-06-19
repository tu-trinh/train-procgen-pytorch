from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from common.constants import ACTION_SPACE, ORIGINAL_ACTION_SPACE

import os, time, yaml, argparse
import gym
from procgen import ProcgenGym3Env
import random
import torch
import pickle
import dill

from PIL import Image
import torchvision as tv

from gym3 import ViewerWrapper, VideoRecorderWrapper, ToBaselinesVecEnv


####################
## HYPERPARAMETERS #
####################
def load_hyperparameters(args):
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[args.param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)
    return hyperparameters

############
## DEVICE ##
############
def set_device(args):
    if args.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using", device)
    if args.device == 'gpu':
        print("GPU set to", os.environ["CUDA_VISIBLE_DEVICES"])
    return device

#################
## ENVIRONMENT ##
#################
def create_venv_render(args, hyperparameters, env_seed, is_valid = False):
    # print('INITIALIZING ENVIRONMENTS...')
    venv = ProcgenGym3Env(
        num = hyperparameters.get("n_envs", 1),
        env_name = args.env_name,
        num_levels = 0 if is_valid else args.num_levels,
        start_level = 0 if is_valid else args.start_level,
        distribution_mode = args.distribution_mode,
        num_threads = 1,
        render_mode = "rgb_array",
        random_percent = args.random_percent,
        corruption_type = args.corruption_type,
        corruption_severity = int(args.corruption_severity),
        continue_after_coin = args.continue_after_coin,
        rand_seed = env_seed
    )
    info_key = None if args.agent_view else "rgb"
    ob_key = "rgb" if args.agent_view else None
    if args.vid_dir is not None:
        venv = VideoRecorderWrapper(
            venv,
            directory = args.vid_dir,
            info_key = info_key,
            ob_key = ob_key,
            fps = args.tps
        )
    venv = ToBaselinesVecEnv(venv)
    venv = VecExtractDictObs(venv, "rgb")
    normalize_rew = hyperparameters.get('normalize_rew', True)
    if normalize_rew:
        venv = VecNormalize(venv, ob=False) # normalizing returns, but not
        #the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


############
## LOGGER ##
############
def set_logger(args):
    # print('INITIALIZING LOGGER...')
    if args.logdir is None:
        logdir = 'procgen/' + args.env_name + '/' + args.exp_name + '/' + 'RENDER_seed' + '_' + \
                 str(args.seed) + '_' + time.strftime("%m-%d-%Y_%H-%M-%S")
        logdir = os.path.join('logs', logdir)
    else:
        logdir = args.logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir_saliency_value = os.path.join(logdir, 'value_saliency')
    if not (os.path.exists(logdir_saliency_value)) and args.value_saliency:
        os.makedirs(logdir_saliency_value)
    logger = Logger(n_envs, logdir)
    return logger

###########
## MODEL ##
###########
def make_model_and_policy(args, env):
    # print('INITIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    if args.reduced_action_space:
        # print("Using reduced action space")
        action_space = gym.spaces.Discrete(len(ACTION_SPACE))
    else:
        # print("Using normal action space")
        action_space = env.action_space
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)
    return model, policy

#############
## STORAGE ##
#############
def set_storage(model, env, n_steps, n_envs, device):
    # print('INITIALIZING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(env.observation_space.shape, hidden_state_dim, n_steps, n_envs, device)
    return storage

###########
## AGENT ##
###########
def make_agent(algo, env, n_envs, policy, logger, storage, device, args,
               all_help_info = [],
               store_percentiles = False,
               all_max_probs = [], all_sampled_probs = [],
               all_max_logits = [], all_sampled_logits = [],
               all_entropies = [],
               probs_by_action = {}, logits_by_action = {}, entropies_by_action = {},
               percentile_dir = None,
               is_expert = False):
    # print('INITIALIZING AGENT...')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, args.num_checkpoints, reduced_action_space = args.reduced_action_space, store_percentiles = store_percentiles, all_max_probs = all_max_probs, all_sampled_probs = all_sampled_probs, all_max_logits = all_max_logits, all_sampled_logits = all_sampled_logits, all_entropies = all_entropies, probs_by_action = probs_by_action, logits_by_action = logits_by_action, entropies_by_action = entropies_by_action, all_help_info = all_help_info, percentile_dir = percentile_dir, by_action = args.by_action, is_expert = is_expert, **hyperparameters)
    if is_expert:
        agent.policy.load_state_dict(torch.load(args.expert_model_file, map_location = device)["model_state_dict"])
    else:
        agent.policy.load_state_dict(torch.load(args.model_file, map_location = device)["model_state_dict"])
    agent.n_envs = n_envs
    return agent

##########
## SAVE ##
##########
def save_run_data(logdir, storage, eval_env_idx, eval_env_seed, as_npy = True):
    run_data = {"help_info_storage": storage.help_info_storage, "action_storage": storage.act_batch.cpu().numpy()}
    with open(logdir + f"/AAA_storage_env_{eval_env_idx}_seed_{eval_env_seed}.pkl", "wb") as f:
        pickle.dump(run_data, f)
    
    """write observations and value estimates to npy / human-readable files"""
    print(f"Saving observations and values to {logdir}")
    if as_npy:
        pass  # FIXME: epoch idx to eval env idx, etc..
        # np.save(logdir + f"/observations_{epoch_idx}", storage.obs_batch)
        # np.save(logdir + f"/value_{epoch_idx}", storage.value_batch)
    else:
        # obs_batch shape: total_steps, num_envs, obs
        values = ""
        actions = ""
        uncertainties = ""
        for step in range(len(storage.obs_batch)):
            for env in range(len(storage.obs_batch[step])):
                obs = storage.obs_batch[step][env]
                o = obs.clone().detach().permute(1, 2, 0)
                o = (o * 255).cpu().numpy().astype(np.uint8)
                Image.fromarray(o).save(logdir + f"/obs_env_{eval_env_idx}_seed_{eval_env_seed}_step_{step}.png")

        #         val = storage.value_batch[step][env]
        #         values += f"Env {env}, step {step}: {val}\n"

        #         try:
        #             act = storage.act_batch[step][env]
        #             if act < 0:
        #                 action = "HELP"
        #             else:
        #                 action = int(act.item())
        #             actions += f"Env {env}, step {step}: {ACTION_MAPPING[action]}\n"
        #         except IndexError:  # last observation has no action
        #             pass

        #         try:
        #             unc = storage.uncertainty_batch[step][env]
        #             unc = float(unc.item())
        #             uncertainties += f"Env {env}, step {step}: {unc}\n"
        #         except IndexError:  # idk
        #             pass
        #     values += "\n"
        #     actions += "\n"
        #     uncertainties += "\n"
        # with open(logdir + f"/values_epoch_{epoch_idx}.txt", "w") as f:
        #     f.write(values)
        # with open(logdir + f"/actions_epoch_{epoch_idx}.txt", "w") as f:
        #     f.write(actions)
        # with open(logdir + f"/uncertainties_epoch_{epoch_idx}.txt", "w") as f:
        #     f.write(uncertainties)
    return

############
## RENDER ##
############
def render(eval_env_idx, eval_env_seed, agent, epochs, args, all_rewards = [], all_adjusted_rewards = [], all_achievement_timesteps = [], all_times_achieved = [], expert = None):
    if args.quant_eval:
        assert epochs == 1, "Just need one epoch for quantitative evaluation"
    if expert is not None:
        assert args.expert_cost is not None and args.switching_cost is not None, "Must have both expert and switching costs"

    obs = agent.env.reset()
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)

    individual_value_idx = 1001
    save_frequency = 1
    saliency_save_idx = 0
    epoch_idx = 0
    cum_reward = 0
    cum_adjusted_reward = 0
    while epoch_idx < epochs:
        agent.policy.eval()
        start_epoch_time = time.time()
        prev_agent = 0  # 0: normal agent action taken, 1: expert action taken
        curr_agent = 0
        for step in range(agent.n_steps):
            if not args.value_saliency:
                act, log_prob_act, value, next_hidden_state, help_info = agent.predict(obs, hidden_state, done, ood_metric = args.ood_metric, risk = args.risk, select_mode = args.select_mode)
                if expert is not None and help_info["need_help"]:
                    act, _, _, _, _ = expert.predict(obs, hidden_state, done, select_mode = args.select_mode)
                    curr_agent = 1
                else:
                    curr_agent = 0
            else:
                act, log_prob_act, value, next_hidden_state, value_saliency_obs = agent.predict_w_value_saliency(obs, hidden_state, done)
                if saliency_save_idx % save_frequency == 0:

                    value_saliency_obs = value_saliency_obs.swapaxes(1, 3)
                    obs_copy = obs.swapaxes(1, 3)

                    ims_grad = value_saliency_obs.mean(axis=-1)

                    percentile = np.percentile(np.abs(ims_grad), 99.9999999)
                    ims_grad = ims_grad.clip(-percentile, percentile) / percentile
                    ims_grad = torch.tensor(ims_grad)
                    blurrer = tv.transforms.GaussianBlur(
                        kernel_size=5,
                        sigma=5.)  # (5, sigma=(5., 6.))
                    ims_grad = blurrer(ims_grad).squeeze().unsqueeze(-1)

                    pos_grads = ims_grad.where(ims_grad > 0.,
                                            torch.zeros_like(ims_grad))
                    neg_grads = ims_grad.where(ims_grad < 0.,
                                            torch.zeros_like(ims_grad)).abs()

                    # Make a couple of copies of the original im for later
                    sample_ims_faint = torch.tensor(obs_copy.mean(-1)) * 0.2
                    sample_ims_faint = torch.stack([sample_ims_faint] * 3, axis=-1)
                    sample_ims_faint = sample_ims_faint * 255
                    sample_ims_faint = sample_ims_faint.clone().detach().type(
                        torch.uint8).cpu().numpy()

                    grad_scale = 9.
                    grad_vid = np.zeros_like(sample_ims_faint)
                    pos_grads = pos_grads * grad_scale * 255
                    neg_grads = neg_grads * grad_scale * 255
                    grad_vid[:, :, :, 2] = pos_grads.squeeze().clone().detach().type(
                        torch.uint8).cpu().numpy()
                    grad_vid[:, :, :, 0] = neg_grads.squeeze().clone().detach().type(
                        torch.uint8).cpu().numpy()

                    grad_vid = grad_vid + sample_ims_faint

                    grad_vid = Image.fromarray(grad_vid.swapaxes(0,2).squeeze())
                    grad_vid.save(logdir_saliency_value + f"/sal_obs_{saliency_save_idx:05d}_grad.png")

                    obs_copy = (obs_copy * 255).astype(np.uint8)
                    obs_copy = Image.fromarray(obs_copy.swapaxes(0,2).squeeze())
                    obs_copy.save(logdir_saliency_value + f"/sal_obs_{saliency_save_idx:05d}_raw.png")

                saliency_save_idx += 1

            next_obs, rew, done, info = agent.env.step(act)
            adjusted_rew = rew.copy()
            if expert is not None and help_info["need_help"]:
                adjusted_rew -= (10 / agent.n_steps) * args.expert_cost  # TODO: reward max will be different once not coinrun
            if curr_agent != prev_agent:
                adjusted_rew -= (10 / agent.n_steps) * args.switching_cost
                # print(f"Step: {step}, needed help? {help_info['need_help']}. Rew = {rew}, adjusted = {adjusted_rew}. Incurred switching cost")
            # else:
                # print(f"Step: {step}, needed help? {help_info['need_help']}. Rew = {rew}, adjusted = {adjusted_rew}. NO switching cost")
            prev_agent = curr_agent
            if args.quant_eval:
                cum_reward += rew
                cum_adjusted_reward += adjusted_rew
            agent.storage.store(obs, hidden_state, act, rew, adjusted_rew, done, info, log_prob_act, value, help_info)
            if all(done):
                break
            obs = next_obs
            hidden_state = next_hidden_state

        _, _, last_val, hidden_state, help_info = agent.predict(obs, hidden_state, done, ood_metric = args.ood_metric, risk = args.risk, select_mode = args.select_mode)
        agent.storage.store_last(obs, hidden_state, last_val, help_info)

        if args.quant_eval:
            all_rewards.append(cum_reward[0])
            all_adjusted_rewards.append(cum_adjusted_reward[0])
            if all(done) and cum_reward[0] > 0:  # use unadjusted rewards as a marker for completion
                all_times_achieved.append(1)
                all_achievement_timesteps.append(step)  # broke out of the while loop on this step, reached the coin
            else:
                all_times_achieved.append(0)
                if step == agent.n_steps - 1:  # simply reached the end without getting the coin
                    all_achievement_timesteps.append(float("inf"))
                else:  # encountered an enemy
                    all_achievement_timesteps.append(-step)
        
        if args.save_run and eval_env_idx % 100 == 0:
            save_run_data(agent.logger.logdir, agent.storage, eval_env_idx, eval_env_seed, args.save_as_npy)

        agent.storage.compute_estimates(agent.gamma, agent.lmbda, agent.use_gae, agent.normalize_adv)
        epoch_idx += 1
        end_epoch_time = time.time()
        # print("[alina] Done with epoch", epoch_idx if not args.save_run else epoch_idx - 1, "took", (end_epoch_time - start_epoch_time) / 60, "minutes")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'render', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'coinrun', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'hard', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'easy-200', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'cpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1), help='number of checkpoints to store')
    parser.add_argument('--logdir',           type=str, default = None)

    #multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    #render parameters
    parser.add_argument('--tps', type=int, default=15, help="env fps")
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--save_run', action='store_true')
    parser.add_argument("--quant_eval", action = "store_true")
    parser.add_argument("--save_as_npy", action="store_true")
    parser.add_argument('--value_saliency', action='store_true')

    parser.add_argument('--random_percent',   type=float, default=0., help='percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--corruption_type',  type=str, default = None)
    parser.add_argument('--corruption_severity',  type=str, default = 1)
    parser.add_argument('--agent_view', action="store_true", help="see what the agent sees")
    parser.add_argument('--continue_after_coin', action="store_true", default = False, help="level doesnt end when agent gets coin")
    parser.add_argument('--noview', action="store_true", help="just take vids")

    parser.add_argument("--store_percentiles", action = "store_true")
    parser.add_argument("--reduced_action_space", action = "store_true")
    parser.add_argument("--ood_metric", type = str, default = None)
    parser.add_argument("--risk", type = int, default = None)
    parser.add_argument("--select_mode", type = str, default = "sample")
    parser.add_argument("--percentile_dir", type = str)
    parser.add_argument("--by_action", action = "store_true")
    parser.add_argument("--expert_model_file", type = str, default = None)
    parser.add_argument("--expert_cost", type = float, default = None)
    parser.add_argument("--switching_cost", type = float, default = None)

    args = parser.parse_args()

    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)

    hyperparameters = load_hyperparameters(args)
    total_envs = hyperparameters.get("total_envs", 100)
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get("n_envs", 1)
    algo = hyperparameters.get('algo', 'ppo')
    epochs = hyperparameters.get("epoch", 1)

    device = set_device(args)
    logger = set_logger(args)

    if args.quant_eval:
        all_rewards = []
        all_adjusted_rewards = []
        all_achievement_timesteps = []
        all_times_achieved = []
        all_help_info = []
    if args.store_percentiles:
        all_max_probs, all_sampled_probs = [], []
        all_max_logits, all_sampled_logits = [], []
        all_entropies = []
        probs_by_action = {i: [] for i in range(len(ACTION_SPACE) if args.reduced_action_space else len(ORIGINAL_ACTION_SPACE))}
        logits_by_action = {i: [] for i in range(len(ACTION_SPACE) if args.reduced_action_space else len(ORIGINAL_ACTION_SPACE))}
        entropies_by_action = {i: [] for i in range(len(ACTION_SPACE) if args.reduced_action_space else len(ORIGINAL_ACTION_SPACE))}
    if args.quant_eval or args.store_percentiles:
        eval_envs = []
        for i in range(total_envs):
            env = create_venv_render(args, hyperparameters, args.seed + i, is_valid = True)
            eval_envs.append(env)
        start = time.time()
        for i, env in enumerate(eval_envs):
            model, policy = make_model_and_policy(args, env)
            storage = set_storage(model, env, n_steps, n_envs, device)
            if args.expert_model_file is not None:
                expert_model, expert_policy = make_model_and_policy(args, env)
                expert_storage = set_storage(expert_model, env, n_steps, n_envs, device)
            if args.quant_eval:
                if args.by_action:
                    help_info = {a: [] for a in range(len(ORIGINAL_ACTION_SPACE))}
                else:
                    help_info = []
                agent = make_agent(algo, env, n_envs, policy, logger, storage, device, args, all_help_info = help_info, percentile_dir = args.percentile_dir)
                if args.expert_model_file is not None:
                    expert_agent = make_agent(algo, env, n_envs, expert_policy, logger, expert_storage, device, args, is_expert = True)
                else:
                    expert_agent = None
                render(i, args.seed + i, agent, epochs, args, all_rewards, all_adjusted_rewards, all_achievement_timesteps, all_times_achieved, expert = expert_agent)
                all_help_info.append(help_info)
            elif args.store_percentiles:
                agent = make_agent(algo, env, n_envs, policy, logger, storage, device, args, store_percentiles = True, all_max_probs = all_max_probs, all_sampled_probs = all_sampled_probs, all_max_logits = all_max_logits, all_sampled_logits = all_sampled_logits, all_entropies = all_entropies, probs_by_action = probs_by_action, logits_by_action = logits_by_action, entropies_by_action = entropies_by_action)
                render(i, args.seed + i, agent, epochs, args)
            if i % 100 == 0:
                end = time.time()
                print("Done with eval", str(i) + ",", "took", (end - start) / 60, "minutes")
                start = time.time()
        if args.quant_eval:
            with open(os.path.join(logger.logdir, f"AAA_quant_eval_{args.model_file.split('/')[-1][:-4]}.txt"), "w") as f:
                f.write(f"Mean reward: {round(np.mean(all_rewards), 3)}\n")
                f.write(f"Median reward: {round(np.median(all_rewards), 3)}\n")
                if args.expert_model_file is not None:
                    f.write(f"Mean adjusted reward: {round(np.mean(all_adjusted_rewards), 3)}\n")
                    f.write(f"Median adjusted reward: {round(np.median(all_adjusted_rewards), 3)}\n")
                try:
                    filtered_achievement_timesteps = [elem for elem in all_achievement_timesteps if elem != float("inf")]
                    f.write(f"Mean timestep achieved: {round(np.mean(filtered_achievement_timesteps))}\n")
                    f.write(f"Median timestep achieved: {round(np.median(filtered_achievement_timesteps))}\n")
                    replaced_achievement_timesteps = []
                    fail_reasons = []  # 0 for being stuck, 1 for dying
                    for elem in all_achievement_timesteps:
                        if elem == float("inf"):  # simply never reached the coin, and run ended
                            replaced_achievement_timesteps.append(n_steps)
                            fail_reasons.append(0)
                        elif elem < 0:  # marker for hitting an enemy at that step
                            replaced_achievement_timesteps.append(-elem)
                            fail_reasons.append(1)
                        else:  # succeeded in reaching coin
                            replaced_achievement_timesteps.append(elem)
                    f.write(f"Mean run length: {round(np.mean(replaced_achievement_timesteps))}\n")
                    f.write(f"Median run length: {round(np.median(replaced_achievement_timesteps))}\n")
                    f.write(f"Proportion of times achieved: {round(np.mean(all_times_achieved), 3)}\n")
                    f.write(f"Proportion of fails due to being stuck: {round(1 - sum(fail_reasons) / len(fail_reasons), 3)}\n")
                    f.write(f"Proportion of fails due to dying: {round(np.mean(fail_reasons), 3)}\n")
                except ValueError:
                    f.write(f"Mean timestep achieved: NONE\n")
                    f.write(f"Median timestep achieved: NONE\n")
                    f.write(f"Mean proportion of times achieved: 0\n")
                    f.write(f"Median proportion of times achieved: 0\n")
                f.write(f"All rewards: {all_rewards}\n\n")
                if args.expert_model_file is not None:
                    f.write(f"All adjusted rewards: {all_adjusted_rewards}\n\n")
                f.write(f"All timesteps: {all_achievement_timesteps}\n\n")
                if args.ood_metric is not None:
                    help_reqs = []
                    for run_help_info in all_help_info:  # for each run
                        reqs = [int(info["need_help"]) for info in run_help_info]  # for each timestep in the run
                        help_reqs.append(reqs)  # all help requests or not in the run
                    f.write(f"Mean times asked for help: {round(np.mean([sum(hr) for hr in help_reqs]))}\n")
                    f.write(f"Median times asked for help: {round(np.median([sum(hr) for hr in help_reqs]))}\n\n")
                    f.write(f"Help times:\n")
                    f.write(str(help_reqs))
        model_suffix = args.model_file.split('/')[-1][:-4]
        if args.store_percentiles:
            with open(os.path.join(logger.logdir, f"all_max_probs_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(all_max_probs, f)
            with open(os.path.join(logger.logdir, f"all_sampled_probs_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(all_sampled_probs, f)
            with open(os.path.join(logger.logdir, f"all_max_logits_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(all_max_logits, f)
            with open(os.path.join(logger.logdir, f"all_sampled_logits_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(all_sampled_logits, f)
            with open(os.path.join(logger.logdir, f"all_entropies_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(all_entropies, f)
            with open(os.path.join(logger.logdir, f"probs_by_action_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(probs_by_action, f)
            with open(os.path.join(logger.logdir, f"logits_by_action_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(logits_by_action, f)
            with open(os.path.join(logger.logdir, f"entropies_by_action_{model_suffix}.pkl"), "wb") as f:
                pickle.dump(entropies_by_action, f)
        print(f"Logging dir:\n{logger.logdir}")
