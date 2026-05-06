import os
import sys
import argparse

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

sys.setrecursionlimit(100000)

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.rlhf.standalone_training import \
    RLHF_finetuning

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    # Create new parser for selector (reward model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector-cfg-file',
                        dest='selector_cfg_file',
                        help='Selector config file path',
                        required=False,
                        default=None,
                        type=str)
    # 提前退出，只执行数据生成，而不进行后续的 DPO 训练
    parser.add_argument('--early-exiting',
                        dest='early_exiting',
                        help="DPO training data generation only",
                        action="store_true")
    # selector_args为上面的参数，extra为其余未定义参数
    selector_args, extra = parser.parse_known_args()

    # Load the LLM config (init_cfg)
    init_cfg = global_cfg.clone()
    args = parse_args(extra)

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    # Indicate this is an RLHF process
    init_cfg.llm.rlhf = True

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # Load the selector config (selector_cfg)
    selector_cfg = global_cfg.clone()
    if selector_args.selector_cfg_file:
        selector_cfg.merge_from_file(selector_args.selector_cfg_file)
    selector_cfg.freeze(save=False)

    init_cfg.freeze()

    # load selector 加载选择器模型与分词器
    selector_backbone_name, _ = selector_cfg.model.type.split('@')
    # load_from_prev_ckpt=True 意味着加载之前联邦学习训练好的检查点
    selector_model = get_llm(selector_cfg,
                             load_from_prev_ckpt=True,
                             device_map='auto')
    selector_tokenizer, _ = get_tokenizer(selector_backbone_name,
                                          selector_cfg.data.root,
                                          selector_cfg.llm.tok_len)

    # load llm 加载待微调的基座模型，这个模型通常是预训练模型或经过SFT的模型，它将被 DPO 算法优化
    model_name, _ = init_cfg.model.type.split('@')
    model = get_llm(init_cfg, device_map='auto')
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root,
                                 init_cfg.llm.tok_len)
    # 对于Decoder-only模型（如 LLaMA, GPT），在进行推理/生成（Generation）时，必须使用左侧填充（Left Padding），否则生成的 Token 位置会错乱。而训练时通常使用右侧填充。因此这里专门加载一个用于生成的 Tokenizer
    generator_tokenizer, _ = get_tokenizer(model_name,
                                           init_cfg.data.root,
                                           init_cfg.llm.tok_len,
                                           padding_side="left")

    # start rlhf training
    gpu_manager = GPUManager(gpu_available=init_cfg.use_gpu,
                             specified_device=init_cfg.device)
    _server_device = gpu_manager.auto_choice()
    RLHF_finetuning(
        model,
        tokenizer,
        init_cfg,
        selector_model,
        selector_tokenizer,
        generator_tokenizer,
        device=_server_device).train(early_exiting=selector_args.early_exiting)
