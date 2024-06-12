import sys
import time
import argparse
import random
import numpy as np
from collections import deque

import funcy as fn
import torch
from tqdm import tqdm

from dfa_sampler import gen_mutated_sequential_reach_avoid
import dfa_embeddings.utils as utils
from dfa_embeddings.dfa_builder import DFABuilder

from dfa_embeddings.dfa_wrapper import DFAEnv
from dfa_embeddings.envs.dummy import DummyEnv

from dfa_embeddings.model import ACModel


import torch_ac
import tensorboardX

class DFA2Vec(object):
    def __init__(self, n_tokens=12, compositional=False, pretrained=True, seed=42):
        super(DFA2Vec, self).__init__()
        self.n_tokens = n_tokens
        self.compositional = compositional
        self.pretrained = pretrained
        self.seed = seed
        self.encoder = None
        self.decoder = None
        self.builder = DFABuilder(n_tokens=self.n_tokens, compositional=self.compositional)
        if pretrained:
            model_name = f"_n_tokens:{self.n_tokens}_seed:{self.seed}_compositional:{self.compositional}"
            storage_dir = "dfa_embeddings/storage"
            model_dir = utils.get_model_dir(model_name, storage_dir)
            try:
                status = utils.get_status(model_dir + "/train")
            except OSError:
                raise Exception("No pretrained model for the given configuration.")
            if "model_state" in status:
                acmodel = ACModel(self.builder.feature_size, self.n_tokens)
                acmodel.load_state_dict(status["model_state"])
                self.encoder = acmodel.gnn
                self.decoder = acmodel.actor
            else:
                raise Exception("Model state is not found for the given configuration.")

    def __call__(self, dfa):
        return self.encoder(self.builder(dfa))

    def train(
        self,
        log_interval=1,
        save_interval=20,
        procs=4,
        frames=10_000_000,
        epochs=2,
        batch_size=1024,
        frames_per_proc=512,
        discount=0.9, lr=0.001,
        gae_lambda=0.5,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        optim_eps=1e-8,
        clip_eps=0.1,
        recurrence=1):

        if self.pretrained:
            raise Exception("This object has already been pretrained.")

        # Set seed for all randomness sources
        utils.seed(self.seed)

        model_name = f"_n_tokens:{self.n_tokens}_seed:{self.seed}_compositional:{self.compositional}"
        storage_dir = "dfa_embeddings/storage"
        model_dir = utils.get_model_dir(model_name, storage_dir)

        # Load loggers and Tensorboard writer
        txt_logger = utils.get_txt_logger(model_dir + "/train", enable_stdout=True)
        csv_file, csv_logger = utils.get_csv_logger(model_dir + "/train")
        tb_writer = tensorboardX.SummaryWriter(model_dir + "/train")

        # Log command and all script arguments
        txt_logger.info("{}\n".format(" ".join(sys.argv)))

        # Set device
        txt_logger.info(f"Device: {utils.device}\n")

        # Load training status
        try:
            status = utils.get_status(model_dir + "/train")
        except OSError:
            status = {"num_frames": 0, "update": 0}
        txt_logger.info("Training status loaded.\n")

        sampler = gen_mutated_sequential_reach_avoid(n_tokens=self.n_tokens)
        if self.compositional:
            sampler = utils.cDFA_sampler(sampler)

        envs = []
        for i in range(procs):
            dummy_env = DummyEnv(n_tokens=self.n_tokens, timeout=75)
            env = DFAEnv(dummy_env, sampler, compositional=self.compositional)
            envs.append(env)

        def preprocessor(obss, device=None):
            return np.array([[self.builder(obs).to(device)] for obs in obss])

        acmodel = ACModel(self.builder.feature_size, self.n_tokens)

        txt_logger.info(f"GAT Number of parameters: {sum(p.numel() for p in acmodel.gnn.parameters() if p.requires_grad)}")
        txt_logger.info(f"embedding size: {acmodel.embedding_size}")

        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
            txt_logger.info("Loading model from existing run.\n")
        acmodel.to(utils.device)
        txt_logger.info("Model loaded.\n")
        txt_logger.info("{}\n".format(acmodel))

        algo = torch_ac.PPOAlgo(envs, acmodel, utils.device, frames_per_proc, discount, lr, gae_lambda,
                                entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                                optim_eps, clip_eps, epochs, batch_size, preprocessor)

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
            txt_logger.info("Loading optimizer from existing run.\n")
        txt_logger.info("Optimizer loaded.\n")

        # Train model

        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()

        while num_frames < frames:
            # Update model parameters

            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % log_interval == 0:
                fps = logs["num_frames"]/(update_end_time - update_start_time)
                duration = int(time.time() - start_time)

                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
                average_discounted_return = utils.average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], discount)
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["average_reward_per_step", "average_discounted_return"]
                data += [average_reward_per_step, average_discounted_return]
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | ARPS: {:.3f} | ADR: {:.3f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status
            if save_interval > 0 and update % save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                          "model_state": algo.acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                utils.save_status(status, model_dir + "/train")
                txt_logger.info("Status saved")

        self.encoder = algo.acmodel.gnn
        self.decoder = algo.acmodel.actor
        
        