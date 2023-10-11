import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import yaml
import time

from torch import nn, optim

from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
from wheeledsim_rl.util.util import dict_to, DummyEnv
from wheeledsim_rl.util.rl_util import split_trajs
from wheeledsim_rl.util.os_util import maybe_mkdir
from wheeledsim_rl.util.normalizer import ObservationNormalizer
from wheeledsim_rl.util.ouNoise import ouNoise
from wheeledsim_rl.policies.ou_policy import OUPolicy
from wheeledsim_rl.collectors.base_collector import Collector
from wheeledsim_rl.trainers.experiment import Experiment

from wheeledsim_rl.networks.str_to_cls import str_to_cls as network_str_to_cls
from wheeledsim_rl.trainers.str_to_cls import str_to_cls as trainer_str_to_cls
from wheeledsim_rl.losses.str_to_cls import str_to_cls as loss_str_to_cls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True, help='The yaml file for the experiment')
    args = parser.parse_args()

    config = yaml.load(open(args.config_fp, 'r'), Loader=yaml.FullLoader)

    sample_traj = torch.load(os.path.join(config['train_fp'], os.listdir(config['train_fp'])[0]))
    env = DummyEnv(sample_traj)

    noise = ouNoise(lowerlimit=env.action_space.low, upperlimit=env.action_space.high, var = np.array([0.01, 0.01]), offset=np.zeros([2, ]), damp=np.ones([2, ])*1e-4)
    policy = OUPolicy(noise)

    print('\nloading train data...')
    train_buf = NStepDictReplayBuffer(env, capacity=3000).to('cuda')
    print('\nloading eval data...')
    
    eval_buf = [torch.load(os.path.join(config['eval_fp'], fp)) for fp in os.listdir(config['eval_fp'])[:50]]

    for fp in os.listdir(config['train_fp'])[:9]:
        traj = torch.load(os.path.join(config['train_fp'], fp))
        train_buf.insert(traj)

    if not isinstance(eval_buf, list):
#        eval_buf = eval_buf.to('cuda')
        trajs = split_trajs(eval_buf.sample_idxs(torch.arange(len(eval_buf))))
    else:
#        trajs = [dict_to(x, 'cuda') for x in eval_buf]
        trajs = eval_buf

    inorm = ObservationNormalizer(env=env)
    onorm = ObservationNormalizer(env=env)

    encoder_dim = config['latent_model']['params']['rnn_hidden_size']
    decoder_dim = encoder_dim + config['latent_model']['params'].get('latent_size', 0)
    batch = train_buf.sample(2, N=3)
    state_dim = batch['observation']['state'].shape[-1]
    act_dim = batch['action'].shape[-1]
    encoders = {}
    decoders = {}
    losses = {}
    #Parse modalitites
    if config['modalities'] is not None:
        for name, info in config['modalities'].items():
            topic = info['topic']
            encoder_params = info['encoder']
            decoder_params = info['decoder']
            loss_params = info['loss']
            
            sample = batch['observation'][topic][:, 0]
            encoder_cls = network_str_to_cls[encoder_params['type']]
            decoder_cls = network_str_to_cls[decoder_params['type']]
            loss_cls = loss_str_to_cls[loss_params['type']]

            #Assumed that the first two net args are insize, outsize
            encoder = encoder_cls(sample.shape[1:], encoder_dim, **encoder_params['params'])
            decoder = decoder_cls(decoder_dim, sample.shape[1:], **decoder_params['params'])
            loss = loss_cls(**loss_params['params'])

            encoders[topic] = encoder
            decoders[topic] = decoder
            losses[topic] = loss

    latent_model_cls = network_str_to_cls[config['latent_model']['type']]
    latent_model = latent_model_cls(encoders, decoders, **config['latent_model']['params'], input_normalizer=inorm, output_normalizer=onorm, state_insize=state_dim, action_insize=act_dim)

    print(latent_model)

    latent_model = latent_model.to('cuda')

    #initialize normalizers
    print('\ninitializing normalizers...')
    tbuf = []

    K=100
    for i in range(K):
        ts = time.time()
        all_data = dict_to(torch.load(os.path.join(config['train_fp'], np.random.choice(os.listdir(config['train_fp'])))), 'cuda')
        tbuf.append(time.time() - ts)
        targets = latent_model.get_training_targets(all_data)
        latent_model.input_normalizer.update(all_data)
        latent_model.output_normalizer.update({'observation':targets})
        print('{}/{}'.format(i+1, K), end='\r')

    print('Avg collect time = {:.6f}s'.format(sum(tbuf)/len(tbuf)))

    print(latent_model.input_normalizer.dump())
    print(latent_model.output_normalizer.dump())
    print('MODEL NPARAMS =', sum([np.prod(x.shape) for x in latent_model.parameters()]))

    opt = optim.Adam(latent_model.parameters(), lr=config['lr'])
    collector = Collector(env, policy=policy)
    collector.to('cuda')

    trainer_cls = trainer_str_to_cls[config['trainer']['type']]

    trainer = trainer_cls(env, policy, latent_model, opt, train_buf, collector, losses=losses, steps_per_epoch=0, eval_dataset=trajs, **config['trainer']['params'])

    experiment = Experiment(trainer, config['name'], config['experiment_fp'], save_logs_every=1, save_every=config['save_every'], train_data_fp=config['train_fp'], buffer_cycle=1)

    maybe_mkdir(os.path.join(config['experiment_fp'], config['name']), force=False)
    with open(os.path.join(config['experiment_fp'], config['name'], '_config.yaml'), 'w') as fp:
        yaml.dump(config, fp)

    experiment.run()
