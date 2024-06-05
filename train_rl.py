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
from dfa_embeddings.dataloader import gen_problems, target_dist
from dfa_embeddings.model import DFAEncoder, ActionPredictor
from dfa_embeddings.model import DFATranformerEncoder
import dfa_embeddings.utils as utils
from dfa_embeddings.dfa_builder import DFABuilder

from dfa_embeddings.dfa_wrapper import DFAEnv
from dfa_embeddings.envs.dummy import DummyEnv

from dfa_embeddings.acmodel import ACModel

import torch_ac
import tensorboardX
    

if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--algo", default="ppo",
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=20,
                        help="number of updates between two saves (default: 20, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=4,
                        help="number of processes (default: 4)")
    parser.add_argument("--frames", type=int, default=10_000_000,
                        help="number of frames of training (default: 10_000_000)")

    # Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of epochs for PPO (default: 2)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size for PPO (default: 1024)")
    parser.add_argument("--frames-per-proc", type=int, default=512,
                        help="number of frames per process before update (default: 512)")
    parser.add_argument("--discount", type=float, default=0.9,
                        help="discount factor (default: 0.9)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.5,
                        help="lambda coefficient in GAE formula (default: 0.5)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.1,
                        help="clipping epsilon for PPO (default: 0.1)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--freeze", action="store_true", default=False, help="Freeze the gradient updates of the DFA module")
    parser.add_argument("--gnn", default="GATv2Conv", help="GATv2Conv | Transformer")
    parser.add_argument("--compositional", action="store_true", default=False, help="Compositional DFAs vs Monolithic DFAs")

    args = parser.parse_args()

    # Set seed for all randomness sources
    utils.seed(args.seed)

    model_name = f"{args.gnn}_seed:{args.seed}_epochs:{args.epochs}_bs:{args.batch_size}_fpp:{args.frames_per_proc}_dsc:{args.discount}_lr:{args.lr}_ent:{args.entropy_coef}_clip:{args.clip_eps}_compositional:{args.compositional}"
    storage_dir = "storage"
    model_dir = utils.get_model_dir(model_name, storage_dir)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir + "/train")
    csv_file, csv_logger = utils.get_csv_logger(model_dir + "/train")
    tb_writer = tensorboardX.SummaryWriter(model_dir + "/train")
    utils.save_config(model_dir + "/train", args)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set device
    txt_logger.info(f"Device: {utils.device}\n")

    # Load training status
    try:
        status = utils.get_status(model_dir + "/train")
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded.\n")

    n_tokens = 12

    sampler = gen_mutated_sequential_reach_avoid(n_tokens=n_tokens)
    if args.compositional:
        sampler = utils.cDFA_sampler(sampler)

    envs = []
    for i in range(args.procs):
        dummy_env = DummyEnv(n_tokens=n_tokens, timeout=75)
        env = DFAEnv(dummy_env, sampler, compositional=args.compositional)
        envs.append(env)

    builder = DFABuilder(n_tokens=n_tokens, compositional=args.compositional)
    def preprocessor(obss, device=None):
        return np.array([[builder(obs).to(device)] for obs in obss])

    acmodel = ACModel(builder.feature_size, n_tokens, args.gnn, args.freeze)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
        txt_logger.info("Loading model from existing run.\n")
    acmodel.to(utils.device)
    txt_logger.info("Model loaded.\n")
    txt_logger.info("{}\n".format(acmodel))

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, utils.device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocessor)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, utils.device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocessor)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Loading optimizer from existing run.\n")
    txt_logger.info("Optimizer loaded.\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)

            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
            average_discounted_return = utils.average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], args.discount)
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
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": algo.acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            utils.save_status(status, model_dir + "/train")
            txt_logger.info("Status saved")

