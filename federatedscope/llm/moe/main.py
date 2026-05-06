import os
import sys
import argparse
import logging
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

file_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(file_dir)

sys.setrecursionlimit(100000)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.llm.moe.init import PeMoEInitializer
from federatedscope.llm.moe.router import PeMoERouter

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector-cfg-file',
                        dest='selector_cfg_file',
                        help='Selector config file path',
                        required=False, 
                        default=None,
                        type=str)
    parser.add_argument('--num-experts', type=int, default=3, help='Number of experts for PeMoE')
    parser.add_argument('--early-exiting', dest='early_exiting', action="store_true")
    
    
    selector_args, extra = parser.parse_known_args()
    args = parse_args(extra)
 
    init_cfg = global_cfg.clone()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    init_cfg.llm.rlhf = True 
    
    # update_logger(init_cfg, clear_before_add=True)
    # setup_seed(init_cfg.seed)
 
    selector_cfg = global_cfg.clone()
    if selector_args.selector_cfg_file:
        selector_cfg.merge_from_file(selector_args.selector_cfg_file)
    
    update_logger(selector_cfg, clear_before_add=True)
    setup_seed(selector_cfg.seed)

    selector_cfg.freeze(save=False)
    
    init_cfg.freeze()

    selector_backbone_name, _ = selector_cfg.model.type.split('@')
    logger.info(f"Loading Selector Model: {selector_backbone_name}")
    selector_model = get_llm(selector_cfg,
                             load_from_prev_ckpt=False,
                             device_map='auto')
    
    selector_tokenizer, _ = get_tokenizer(selector_backbone_name,
                                          selector_cfg.data.root,
                                          selector_cfg.llm.tok_len)

    model_name, _ = init_cfg.model.type.split('@')
    logger.info(f"Loading Generator Model: {model_name}")
    model = get_llm(init_cfg, device_map='auto')
    
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root, init_cfg.llm.tok_len)
    generator_tokenizer, _ = get_tokenizer(model_name,
                                           init_cfg.data.root,
                                           init_cfg.llm.tok_len,
                                           padding_side="left")

    gpu_manager = GPUManager(gpu_available=init_cfg.use_gpu,
                             specified_device=init_cfg.device)
    _server_device = gpu_manager.auto_choice()
    
    
    initializer = PeMoEInitializer(
        model=model,                       # Generator
        tokenizer=tokenizer,
        config=init_cfg,
        selector_model=selector_model,     # Selector (Experts Base)
        selector_tokenizer=selector_tokenizer,
        generator_tokenizer=generator_tokenizer,
        device=_server_device,
        selector_cfg = selector_cfg
    )

    
    trained_selector, trained_router, embeddings_shape, router_save_path = initializer.run_initialization()

    selector_cfg.defrost()
    selector_cfg.llm.hidden_size = embeddings_shape
    selector_cfg.federate.save_to_router = router_save_path
    data, modified_cfg = get_data(config=selector_cfg.clone(), init_cfg=init_cfg) 
    selector_cfg.merge_from_other_cfg(modified_cfg)

    
    if selector_cfg.federate.client_idx_for_local_train != 0: 
        selector_cfg.federate.client_num = 1  
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[selector_cfg.federate.client_idx_for_local_train] 
        data = new_data
    selector_cfg.freeze()


    runner = get_runner(data=data,  
                        server_class=get_server_cls(selector_cfg),
                        client_class=get_client_cls(selector_cfg),
                        config=selector_cfg.clone(),
                        client_configs=None)
    _ = runner.run() 