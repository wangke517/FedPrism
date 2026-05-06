import os
import sys

DEV_MODE = False  # simplify the federatedscope re-setup everytime we change
# the source codes of federatedscope
# 允许开发者在没有使用pip install安装FederatedScope包的情况下，直接运行源码进行调试。
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir) # 将上级目录加入系统路径

sys.setrecursionlimit(100000) # 修改Python允许的最大递归深度，避免递归过深时报错

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

# 强制删除了环境变量中的HTTP/HTTPS代理设置
# 在联邦学习（特别是分布式或本地模拟通信）中，代理常常会导致不同进程（Client 和 Server）之间的通信连接失败
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone() # 一个包含各种针对config函数的全局默认配置，不直接修改全局的默认配置，而是复制一份副本，防止配置被意外污染
    args = parse_args() # 获取命令行输入的参数
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file) # 读取指定的YAML文件，并用文件里的设置覆盖默认设置
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts) # 将通用的配置参数和特定于某个客户端的配置参数分离开
    init_cfg.merge_from_list(cfg_opt) # 允许在命令行临时修改配置。例如 python main.py ... federate.total_round_num 200，这里的200会覆盖YAML文件里的设置

    update_logger(init_cfg, clear_before_add=True) # 配置日志输出格式和路径
    setup_seed(init_cfg.seed) # 固定随机种子，确保实验可复现

    # load clients' cfg file 加载客户端特定配置
    # 如果有client_cfg_file，则加载这份特定的配置，用于个性化设置每个Client
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt) # 命令行的配置覆盖原配置
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone(), # 下载并预处理数据
                                  client_cfgs=client_cfgs) # 有些配置在加载数据前是未知的。例如：“这个数据集到底有多少个类别？” get_data 会计算出真实的类别数等信息
    init_cfg.merge_from_other_cfg(modified_cfg) # 并返回modified_cfg，然后回填到主配置init_cfg 中。这实现了配置的自适应

    if init_cfg.federate.client_idx_for_local_train != 0: # 用于调试，如果设置了 client_idx_for_local_train（非0）
        init_cfg.federate.client_num = 1  # 框架会丢弃其他所有客户端，假装只有这一个客户端，主要用于快速调试本地模型代码，而不需要跑完整个联邦流程
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[init_cfg.federate.client_idx_for_local_train] # 重组data字典，只保留指定的客户端数据
        data = new_data

    init_cfg.freeze() # 锁定配置，防止后续代码意外修改

    runner = get_runner(data=data,  # 根据配置（是 Standalone 还是 Distributed）返回对应的Runner实例
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run() # 启动！！！
