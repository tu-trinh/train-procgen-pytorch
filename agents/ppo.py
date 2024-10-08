from .base_agent import BaseAgent
from common.misc_util import adjust_lr
from common.constants import *
from common.hasher import HashSet
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pickle
import json
import h5py
import os
import sys
sys.path.append("/Users/tutrinh/Work/CHAI/")
sys.path.append("/nas/ucb/tutrinh/")
sys.path.append("/Users/tutrinh/Work/CHAI/yield_request_control/")
sys.path.append("/nas/ucb/tutrinh/yield_request_control/")
from yield_request_control.detector.dataset import global_contrast_normalization, standardization, preprocess_and_save_images, get_datasets
from yield_request_control.detector.trainers import dynamic_permute
from yield_request_control.detector.deep_svdd import DeepSVDD


class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 save_timesteps=None,
                 save_observations=False,
                 concurrent_svdd=False,
                 reduced_action_space=False,
                 store_percentiles=False,
                 all_sampled_probs=[],
                 all_max_probs=[],
                 all_sampled_logits=[],
                 all_max_logits=[],
                 all_entropies=[],
                 probs_by_action={},
                 logits_by_action={},
                 entropies_by_action={},
                 all_help_info=[],
                 percentile_dir=None,
                 detector_model=None,
                 use_latent=False,
                 is_expert=False,
                 by_action=False,
                 unique_actions=False,
                 env_valid=None,
                 storage_valid=None,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device,
                                  n_checkpoints, save_timesteps, env_valid, storage_valid)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.reduced_action_space = reduced_action_space
        self.store_percentiles = store_percentiles
        self.save_observations = save_observations
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

        if self.store_percentiles:
            self.all_max_probs = all_max_probs
            self.all_sampled_probs = all_sampled_probs
            self.all_max_logits = all_max_logits
            self.all_sampled_logits = all_sampled_logits
            self.all_entropies = all_entropies
            self.probs_by_action = probs_by_action
            self.logits_by_action = logits_by_action
            self.entropies_by_action = entropies_by_action

        self.request_limit = 3
        self.num_requests = 0
        self.num_actions = env.action_space.n
        if detector_model is None and percentile_dir is not None:
            self.by_action = by_action
            if self.reduced_action_space:
                pass
            else:
                with open(os.path.join(percentile_dir, "percentiles.pkl"), "rb") as f:
                    percentiles = pickle.load(f)
                self.max_probability_thresholds = percentiles["max_probs"]
                self.sampled_probability_thresholds = percentiles["sampled_probs"]
                self.max_logit_thresholds = percentiles["max_logits"]
                self.sampled_logit_thresholds = percentiles["sampled_logits"]
                self.entropy_thresholds = percentiles["entropies"]
                # self.probability_thresholds_by_action = percentiles["entropies_by_action"]
                # self.logit_thresholds_by_action = percentiles["logits_by_action"]
                # self.entropy_thresholds_by_action = percentiles["probs_by_action"]
        self.detector_model = detector_model
        self.use_latent = use_latent
        self.concurrent_svdd = concurrent_svdd
        if self.concurrent_svdd:
            self.detector_model = DeepSVDD(for_latent = self.use_latent)
            self.detector_model.set_network("concurrent")
            self.detector_model.save_model(os.path.join(self.logger.logdir, "network.tar"), os.path.join(self.logger.logdir, "autoencoder.tar"))
        if self.detector_model is not None:
            if use_latent:
                self.transforms = transforms.Compose([
                    transforms.Lambda(lambda x: standardization(x.squeeze()))
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Lambda(lambda x: global_contrast_normalization(x, scale = "l1")),
                    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
                ])
            if not self.concurrent_svdd:  # if detector is initialized but not concurrently training, then it must be testing
                with open(os.path.join(percentile_dir, "results.json"), "r") as f:
                    detector_results = json.load(f)
                self.detector_thresholds = {int(k): v for k, v in detector_results["test_thresholds"].items()}  # risk is pseudo-percentiles from 50 to 150
        self.all_help_info = all_help_info
        self.is_expert = is_expert

        self.unique_actions = unique_actions
        if self.unique_actions:
            self.state_action_tracker = HashSet()

    def determine_ask_for_help(self, obs, latent_rep, metric, risk, act, dist, logits):
        if metric == "msp":
            need_help = torch.log(dist.probs.max()) < np.log(self.max_probability_thresholds[risk])
        elif metric == "sampled_p":
            if self.by_action:
                need_help = dist.log_prob(act) < np.log(self.probability_thresholds_by_action[act.item()][risk])
            else:
                need_help = dist.log_prob(act) < np.log(self.sampled_probability_thresholds[risk])
        elif metric == "ml":
            need_help = logits.max() < self.max_logit_thresholds[risk]
        elif metric == "sampled_l":
            if self.by_action:
                need_help = logits.squeeze()[act] < self.logit_thresholds_by_action[act.item()][risk]
            else:
                need_help = logits.squeeze()[act] < self.sampled_logit_thresholds[risk]
        elif metric == "ent":
            if self.by_action:
                need_help = dist.entropy() > self.entropy_thresholds_by_action[act.item()][100 - risk]
            else:
                need_help = dist.entropy() > self.entropy_thresholds[100 - risk]
        elif metric == "random":
            need_help = np.random.random() < risk / 100
        elif metric == "detector":
            if self.use_latent:
                inputs = self.transforms(latent_rep)
            else:
                inputs = self.transforms(obs)
            inputs = dynamic_permute(inputs.float().to(self.device))
            if inputs.size(0) != 1:
                inputs = inputs.unsqueeze(0)
            outputs = self.detector_model.net(inputs)
            distance = torch.sum((outputs - self.detector_model.center)**2, dim = 1)
            need_help = distance.item() > self.detector_thresholds[risk]
        help_info = {}
        sorted_probs, sorted_indices = torch.sort(dist.probs, descending = True)
        sorted_probs = sorted_probs.squeeze()
        sorted_indices = sorted_indices.squeeze()
        sorted_logits = logits.squeeze()[sorted_indices]
        # action_info is a list of tuples (action name, probability, logit)
        if self.reduced_action_space:
            mapping = ACTION_MAPPING
        else:
            mapping = ORIGINAL_ACTION_MAPPING
        action_info = [(mapping[act.item()], dist.probs.squeeze()[act.item()].item(), logits.squeeze()[act.item()].item())]
        for idx in sorted_indices:
            if idx.item() != act.item():
                action_info.append((mapping[idx.item()], sorted_probs[idx.item()].item(), sorted_logits[idx.item()].item()))
        help_info["action_info"] = action_info
        help_info["entropy"] = dist.entropy().item()
        try:
            help_info["distance"] = distance.item()
        except UnboundLocalError:  # not doing svdd => no distance
            pass
        help_info["need_help"] = need_help.item() if isinstance(need_help, torch.Tensor) else need_help
        self.all_help_info.append(help_info)
        return need_help, help_info
    
    def predict(self, obs, hidden_state, done, ood_metric = None, risk = None, select_mode = "sample"):
        assert ood_metric in [None, "msp", "ml", "sampled_p", "sampled_l", "ent", "random", "detector"], "Check ood metric"
        assert select_mode in ["sample", "max"], "Check select mode"
        if ood_metric is not None:
            assert risk is not None, "Must provide risk for ood metric"
        if isinstance(done, list):
            done = torch.tensor(done).float()
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, logits, value, hidden_state, latent_rep = self.policy(obs, hidden_state, mask)
            if ood_metric is None or self.is_expert or not self.unique_actions:  # likely training phase, or this is the expert, or we don't care about repeated actions in repeat states
                if select_mode == "sample":
                    act = dist.sample()
                else:
                    act = torch.argmax(dist.probs).unsqueeze(0)
                repeated_state = False
            else:  # likely evaluation phase
                if self.state_action_tracker.has_seen_key(obs):
                    repeated_state = True
                    seen_actions = self.state_action_tracker.get_vals(obs)
                    unseen_mask = torch.ones_like(dist.probs).bool()
                    unseen_mask[:, list(seen_actions)] = False
                    unseen_probs = dist.probs.clone()
                    if select_mode == "sample":
                        if all(~unseen_mask[0]):
                            self.state_action_tracker.reset(obs)
                            act = dist.sample()
                        else:
                            unseen_probs[~unseen_mask] = 0
                            unseen_probs /= unseen_probs.sum()
                            unseen_dist = torch.distributions.Categorical(probs = unseen_probs)
                            act = unseen_dist.sample()
                    else:
                        if all(~unseen_mask[0]):
                            self.state_action_tracker.reset(obs)
                            act = torch.argmax(dist.probs).unsqueeze(0)
                        else:
                            unseen_probs[~unseen_mask] = float("-inf")
                            act = torch.argmax(unseen_probs).unsqueeze(0)
                else:
                    repeated_state = False
                    if select_mode == "sample":
                        act = dist.sample()
                    else:
                        act = torch.argmax(dist.probs).unsqueeze(0)
                self.state_action_tracker.add_val(obs, act)
            log_prob_act = dist.log_prob(act)
            if self.store_percentiles:
                action_probs = dist.probs.gather(1, act.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                action_logits = logits.gather(1, act.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                entropies = dist.entropy().cpu().numpy()
                self.all_max_probs.extend(dist.probs.max(dim = -1)[0].cpu().numpy())
                self.all_sampled_probs.extend(action_probs)
                self.all_max_logits.extend(logits.max(dim = -1)[0].cpu().numpy())
                self.all_sampled_logits.extend(action_logits)
                self.all_entropies.extend(entropies)
                for i in range(act.shape[0]):
                    self.probs_by_action[act.cpu().numpy()[i]].append(action_probs[i])
                    self.logits_by_action[act.cpu().numpy()[i]].append(action_logits[i])
                    self.entropies_by_action[act.cpu().numpy()[i]].append(entropies[i])
        if not self.is_expert and ood_metric is not None:
            need_help, help_info = self.determine_ask_for_help(obs, latent_rep, ood_metric, risk, act, dist, logits)
            if need_help:
                self.num_requests += 1
        else:
            help_info = None
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy(), latent_rep.cpu().numpy(), help_info, repeated_state

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1-done).to(device=self.device)
        dist, logits, value, hidden_state, latent_rep = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), hidden_state.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, logits_batch, value_batch, _, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def train(self, num_timesteps, reduced_action_space):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        save_timestep_index = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        if self.save_observations:
            this_run_obs = []
            obs_set_num = 0
        if self.concurrent_svdd:
            copied_weights = False

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state, latent_rep, help_info, _ = self.predict(obs, hidden_state, done, ood_metric = None)
                if reduced_action_space:
                    act = ACTION_TRANSLATION[act]
                    assert act.shape == log_prob_act.shape, "Messed up converting actions"
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, latent_rep, hidden_state, act, rew, rew.copy(), False, False, done, info, log_prob_act, value, help_info, 0)
                if self.save_observations:  # obs is a numpy array
                    this_run_obs.append(obs)
                obs = next_obs
                hidden_state = next_hidden_state
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val, hidden_state, latent_rep, help_info, _ = self.predict(obs, hidden_state, done, ood_metric = None)
            if self.save_observations:
                this_run_obs.append(obs)
                # with h5py.File(os.path.join(self.logger.logdir, "saved_obs.h5"), "w" if self.t < 1 else "a") as f:
                #     for obs_idx, saved_obs in enumerate(this_run_obs):
                #         f.create_dataset(f"run_{obs_set_num}_obs_{obs_idx}", data = saved_obs, compression = "gzip", compression_opts = 9)
                if self.concurrent_svdd:
                    obs_save_path = os.path.join(self.logger.logdir, f"saved_obs.npz")
                else:
                    obs_save_path = os.path.join(self.logger.logdir, f"saved_obs_run_{obs_set_num}.npz")
                np.savez_compressed(obs_save_path, np.array(this_run_obs))
                obs_set_num += 1
                this_run_obs = []
                if self.concurrent_svdd:
                    preprocess_and_save_images(self.logger.logdir, self.logger.logdir, "np", save_unique = False)
            self.storage.store_last(obs, latent_rep, hidden_state, last_val, help_info, 0)
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            #valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, log_prob_act_v, value_v, next_hidden_state_v, latent_rep_v, help_info, _ = self.predict(obs_v, hidden_state_v, done_v, ood_metric = None)
                    if reduced_action_space:
                        act = ACTION_TRANSLATION[act]
                        assert act.shape == log_prob_act.shape, "Messed up converting actions (val)"
                    next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act)
                    self.storage_valid.store(obs_v, latent_rep_v, hidden_state_v, act_v,
                                             rew_v, rew_v.copy(), False, False, done_v, info_v,
                                             log_prob_act_v, value_v, help_info, 0)
                    obs_v = next_obs_v
                    hidden_state_v = next_hidden_state_v
                _, _, last_val_v, hidden_state_v, latent_rep_v, help_info, _ = self.predict(obs_v, hidden_state_v, done_v, ood_metric = None)
                self.storage_valid.store_last(obs_v, latent_rep_v, hidden_state_v, last_val_v, help_info, 0)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            
            # Save the model
            if self.use_save_intervals:
                if self.t > ((checkpoint_cnt+1) * save_every):
                    print("Saving model.")
                    torch.save({'model_state_dict': self.policy.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()},
                                self.logger.logdir + '/model_' + str(self.t) + '.pth')
                    checkpoint_cnt += 1
            else:
                try:
                    if self.t + 1 >= self.save_timesteps[save_timestep_index]:
                        print("Saving model at timestep", self.t + 1)
                        torch.save({'model_state_dict': self.policy.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()},
                                    self.logger.logdir + '/model_' + str(self.t) + '.pth')
                        save_timestep_index += 1
                except IndexError:  # no more timesteps needed to be saved
                    pass
            
            # Load SVDD model, (pre)train it, and save it back
            if self.concurrent_svdd:
                self.detector_model.load_model(os.path.join(self.logger.logdir, "network.tar"), os.path.join(self.logger.logdir, "autoencoder.tar"))
                train_dataset, _ = get_datasets(self.logger.logdir, False)
                if self.t <= 0.2 * num_timesteps:
                    self.detector_model.pretrain(train_dataset, None, num_epochs = 1, batch_size = 128)
                else:
                    if not copied_weights:
                        self.detector_model.init_network_weights()
                        copied_weights = True
                    self.detector_model.train(train_dataset, None, num_epochs = 1, batch_size = 128)
                self.detector_model.save_model(os.path.join(self.logger.logdir, "network.tar"), os.path.join(self.logger.logdir, "autoencoder.tar"))

        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()
        # if self.store_percentiles:
        #     with open(self.logger.logdir + "/all_probs.pkl", "wb") as f:
        #         pickle.dump(self.all_probs, f)
        #     with open(self.logger.logdir + "/all_logits.pkl", "wb") as f:
        #         pickle.dump(self.all_logits, f)
        #     with open(self.logger.logdir + "/probs_by_action.pkl", "wb") as f:
        #         pickle.dump(self.probs_by_action, f)
        #     with open(self.logger.logdir + "/logits_by_action.pkl", "wb") as f:
        #         pickle.dump(self.logits_by_action, f)

