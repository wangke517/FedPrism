import abc
import logging

from collections import deque
import heapq

import numpy as np

from federatedscope.core.workers import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import get_resource_info, \
    get_ds_rank
from federatedscope.core.auxiliaries.feat_engr_builder import \
    get_feat_engr_wrapper

logger = logging.getLogger(__name__)


class BaseRunner(object):
    """
    This class is a base class to construct an FL course, which includes \
    ``_set_up()`` and ``run()``.

    Args:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Attributes:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server: The instantiated server.
        client: The instantiate client(s).
        cfg : The configurations of the FL course.
        client_cfgs: The clients' configurations.
        mode: The run mode for FL, ``distributed`` or ``standalone``
        gpu_manager: manager of GPU resource
        resource_info: information of resource
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using Runner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs
        self.serial_num_for_msg = 0

        self.mode = self.cfg.federate.mode.lower() # GPU训练的方式（standalone）
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        self.unseen_clients_id = []
        self.feat_engr_wrapper_client, self.feat_engr_wrapper_server = \
            get_feat_engr_wrapper(config)
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check the completeness of msg_handler.
        self.check()

        # Set up for Runner
        self._set_up()

    @abc.abstractmethod
    def _set_up(self):
        """
        Set up and instantiate the client/server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_server_args(self, resource_info, client_resource_info):
        """
        Get the args for instantiating the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            (server_data, model, kw): None or data which server holds; model \
            to be aggregated; kwargs dict to instantiate the server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_client_args(self, client_id, resource_info):
        """
        Get the args for instantiating the server.

        Args:
            client_id: ID of client
            resource_info: information of resource

        Returns:
            (client_data, kw): data which client holds; kwargs dict to \
            instantiate the client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        Launch the FL course

        Returns:
            dict: best results during the FL course
        """
        raise NotImplementedError

    @property
    def ds_rank(self):
        return get_ds_rank()

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up and instantiate the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            Instantiate server.
        """
        # --- 1. 类检查 ---
        # 确保传入的 server_class 不为空，这是实例化对象的基础
        assert self.server_class is not None, \
            "`server_class` cannot be None."
        # --- 2. ID 设置 ---
        # 在 FederatedScope 中，Server 的 ID 默认固定为 0
        self.server_id = 0
        # --- 3. 参数准备 ---
        # 调用抽象方法 _get_server_args 获取实例化所需的特定参数
        # 返回值包括：
        # - server_data: Server 端的数据（如果有）
        # - model: 全局模型架构
        # - kw: 额外的关键字参数（如 Standalone 模式下的 shared_comm_queue）
        server_data, model, kw = self._get_server_args(resource_info,
                                                       client_resource_info)
        # --- 4. 设备分配 ---
        # 使用 GPUManager 自动选择一个可用的设备（GPU 或 CPU）给 Server
        self._server_device = self.gpu_manager.auto_choice()
        # --- 5. 实例化 Server ---
        # 调用 Server 类（或其子类）的构造函数 (这里指Server或者LLMMultiLoRAServer)
        server = self.server_class(
            ID=self.server_id,
            config=self.cfg,
            data=server_data,
            model=model,
            client_num=self.cfg.federate.client_num,
            total_round_num=self.cfg.federate.total_round_num,
            device=self._server_device,
            unseen_clients_id=self.unseen_clients_id, # 未见客户端（用于泛化测试）
            **kw) # 展开额外的参数
        # --- 6. 功能增强（Wrappers）---
        # 这一部分展示了框架的模块化设计，通过装饰器模式动态增强 Server 功能
        # 6.1 NBAFL (No-Box Adversarial FL) 支持
        if self.cfg.nbafl.use:
            from federatedscope.core.trainers.trainer_nbafl import \
                wrap_nbafl_server
            wrap_nbafl_server(server)
        # 6.2 纵向联邦 (Vertical FL) 支持
        if self.cfg.vertical.use:
            from federatedscope.vertical_fl.utils import wrap_vertical_server
            server = wrap_vertical_server(server, self.cfg)
        # 6.3 SWA (Stochastic Weight Averaging) 支持
        # 随机权重平均，一种提升模型泛化能力的聚合策略
        if self.cfg.fedswa.use:
            from federatedscope.core.workers.wrapper import wrap_swa_server
            server = wrap_swa_server(server)
        # --- 7. 日志与返回 ---
        logger.info('Server has been set up ... ')
        # 应用特征工程包装器（如果配置了相关的特征处理逻辑）
        return self.feat_engr_wrapper_server(server)

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up and instantiate the client.

        Args:
            client_id: ID of client
            client_model: model of client
            resource_info: information of resource

        Returns:
            Instantiate client.
        """
        # --- 1. 类检查 ---
        assert self.client_class is not None, \
            "`client_class` cannot be None"
        # 这里再次设置 server_id=0，主要是为了让 Client 知道 Server 的 ID
        self.server_id = 0
        # --- 2. 参数准备 ---
        # 获取该 Client 独有的数据和额外参数
        client_data, kw = self._get_client_args(client_id, resource_info)
        # --- 3. 客户端异构配置 (Client-Specific Config) ---
        # 这一步非常关键，支持了"系统异构"或"模型异构"
        # 首先克隆全局配置，防止修改影响其他 Client
        client_specific_config = self.cfg.clone()
        if self.client_cfgs: # 如果存在针对特定 Client 的配置 (client_cfgs)
            client_specific_config.defrost()  # 解锁配置以允许修改
            # 将该 Client (client_{id}) 的特定配置合并进来
            # 例如：Client 1 的学习率是 0.01，Client 2 的学习率是 0.001
            client_specific_config.merge_from_other_cfg(
                self.client_cfgs.get('client_{}'.format(client_id)))
            client_specific_config.freeze() # 重新锁定
        # --- 4. 设备分配逻辑 ---
        # 这里有一个针对 Standalone 模式的优化逻辑：
        # 如果开启了 share_local_model (共享模型对象以节省内存)，
        # 强制 Client 使用和 Server 相同的设备 (self._server_device)。
        # 否则，模型会在不同 GPU/CPU 之间来回拷贝，导致性能极差。
        client_device = self._server_device if \
            self.cfg.federate.share_local_model else \
            self.gpu_manager.auto_choice()
        # --- 5. 实例化 Client ---
        client = self.client_class(
            ID=client_id,
            server_id=self.server_id,
            config=client_specific_config,
            data=client_data,
            # 模型处理逻辑：
            # 如果传入了 client_model (共享模型)，直接使用；
            # 否则，调用 get_model 为该 Client 创建一个新的模型实例。
            model=client_model or get_model(
                client_specific_config, client_data, backend=self.cfg.backend),
            device=client_device,
            is_unseen_client=client_id in self.unseen_clients_id, # 是否为未见客户端
            **kw) # 展开额外参数

        if self.cfg.vertical.use:
            from federatedscope.vertical_fl.utils import wrap_vertical_client
            client = wrap_vertical_client(client, config=self.cfg)

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return self.feat_engr_wrapper_client(client)

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return


class StandaloneRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server and client for standalone mode.
        """
        self.is_run_online = True if self.cfg.federate.online_aggr else False # 检查是否开启了"在线聚合" (Online Aggregation) 模式
        self.shared_comm_queue = deque() # 【核心机制】初始化一个双端队列，作为唯一的"虚拟信道"，所有Client发出的消息和Server发出的指令都存放在这里

        if self.cfg.backend == 'torch': # 如果后端是PyTorch，强制将线程数设为 1
            import torch
            torch.set_num_threads(1)  # 原因：在单机模拟大量客户端（如100个）时，多线程并行会导致CPU争抢和死锁

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"  # 安全检查：必须有客户端参与

        if self.cfg.federate.method == "global":  # 特殊模式：Global (集中式训练)
            self.cfg.defrost()  # 如果方法是global，强制把客户端数量设为1，模拟只有一台机器拥有所有数据
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:  # 用于模拟系统异构（System Heterogeneity），给不同客户端分配不同的虚拟硬件资源
            if len(self.resource_info) < self.cfg.federate.client_num + 1:  # 如果提供的资源配置样本不够分，则允许重复采样 (replace=True)
                replace = True # 允许随机抽样
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num + 1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(  # 随机分配资源ID
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(  # 调用父类BaseRunner的方法初始化Server实例
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()
        # assume the client-wise data are consistent in their input&output
        # shape
        # --- 模型内存优化 (Model Sharing) ---
        # 如果开启 share_local_model，所有客户端共用同一个模型对象，节省内存
        if self.cfg.federate.online_aggr: # 在线聚合模式下，根据第1个客户端的数据初始化模型
            self._shared_client_model = get_model(
                self.cfg, self.data[1], backend=self.cfg.backend
            ) if self.cfg.federate.share_local_model else None
        else: # 普通模式下，客户端直接共享Server的模型对象（作为初始模型）
            self._shared_client_model = self.server.model \
                if self.cfg.federate.share_local_model else None
        # 循环创建所有Client实例
        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model, # 这里传入了client模型实例或者none
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

        # in standalone mode, by default, we print the trainer info only
        # once for better logs readability # 打印第1个Client的训练器信息，避免打印几百遍刷屏
        trainer_representative = self.client[1].trainer
        if trainer_representative is not None and hasattr(
                trainer_representative, 'print_trainer_meta_info'):
            trainer_representative.print_trainer_meta_info()

    # 负责打包参数，供 _setup_server调用
    def _get_server_args(self, resource_info=None, client_resource_info=None):
        if self.server_id in self.data:  # 确定Server是否拥有数据
            server_data = self.data[self.server_id]
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)  # 根据Server数据建立模型
        else:
            server_data = None # 如果Server没数据，借用Client 1的数据来确定模型输入输出维度
            data_representative = self.data[1]
            model = get_model(
                self.cfg, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        kw = {
            'shared_comm_queue': self.shared_comm_queue, # 【关键】将 shared_comm_queue 塞入参数，确保 Server 能访问公共通信队列
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'shared_comm_queue': self.shared_comm_queue, # 【关键】将 shared_comm_queue 塞入参数，确保 Client 能访问公共通信队列
            'resource_info': resource_info
        }
        return client_data, kw

    def run(self): # 启动仿真的总指挥
        for each_client in self.client:
            # Launch each client
            self.client[each_client].join_in() # 让每个客户端发送 "JoinIn" 消息到队列，申请加入训练

        if self.is_run_online:
            self._run_simulation_online() # 特殊模式
        else:
            self._run_simulation() # 【标准模式】进入主消息循环
        # TODO: avoid using private attr 训练结束，通知监控器停止记录
        self.server._monitor.finish_fed_runner(fl_mode=self.mode)
        return self.server.best_results # 返回服务器记录的最佳结果

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the \
        standalone mode)
        """
        if rcv != -1: # 逻辑分支1：如果指定了 rcv (接收者ID)，说明这是一次模拟的广播中的单点发送
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        # 逻辑分支2：标准路由
        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes() # 统计流量
        if not isinstance(receiver, list): # 确保接收者是列表格式
            receiver = [receiver]
        for each_receiver in receiver: # 遍历所有接收者进行分发
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg) # 如果接收者是 0，交给 Server 处理
                self.server._monitor.track_download_bytes(download_bytes) # Server 记录下行流量
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg) # 如果接收者是 >0，交给对应的 Client 处理
                self.client[each_receiver]._monitor.track_download_bytes( # Client 记录下行流量
                    download_bytes)

    def _run_simulation_online(self):
        """
        Run for online aggregation.
        Any broadcast operation would be executed client-by-clien to avoid \
        the existence of #clients messages at the same time. Currently, \
        only consider centralized topology \
        """
        def is_broadcast(msg):
            return len(msg.receiver) >= 1 and msg.sender == 0

        cached_bc_msgs = []
        cur_idx = 0
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if is_broadcast(msg):
                    cached_bc_msgs.append(msg)
                    # assume there is at least one client
                    msg = cached_bc_msgs[0]
                    self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                    cur_idx += 1
                    if cur_idx >= len(msg.receiver):
                        del cached_bc_msgs[0]
                        cur_idx = 0
                else:
                    self._handle_msg(msg)
            elif len(cached_bc_msgs) > 0:
                msg = cached_bc_msgs[0]
                self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                cur_idx += 1
                if cur_idx >= len(msg.receiver):
                    del cached_bc_msgs[0]
                    cur_idx = 0
            else:
                # finished
                break

    def _run_simulation(self):
        """
        Run for standalone simulation (W/O online aggr)
        """
        server_msg_cache = list()  # 用于 Server 的消息缓存堆（Heap），用于按时间戳排序
        while True: # 死循环，直到没有消息处理为止
            # --- 阶段1：处理公共队列 (Immediate Queue) ---
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft() # 取出一条消息
                if not self.cfg.vertical.use and msg.receiver == [ # 如果不是纵向联邦 且接收者只有Server
                        self.server_id
                ]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    # 【关键】Server 不立即处理消息，而是放入堆中
                    # 这是为了模拟网络延迟或处理顺序：Server 应该先处理timestamp早的消息
                    msg.serial_num = self.serial_num_for_msg # msg在本self的序号
                    self.serial_num_for_msg += 1
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            # --- 阶段2：Server 处理缓存 (Cached Queue) ---
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache) # 当公共队列空了，Server 开始从自己的缓存堆里拿消息（时间戳最小的先出）
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up': # 处理异步联邦学习 (Asynchronous FL) 的超时逻辑
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp): # 如果消息时间戳超过了当前轮次的截止时间
                        heapq.heappush(server_msg_cache, msg) # 触发超时事件，消息放回堆里等待下一轮处理
                    else:
                        self._handle_msg(msg) # 正常处理
                else:
                    self._handle_msg(msg) # 正常同步模式，直接处理
            # --- 阶段3：退出条件 ---
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up': # 异步模式最后的清理
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break # 同步模式直接退出


class DistributedRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server or client for distributed mode.
        """
        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port + self.ds_rank
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port + self.ds_rank
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def _get_server_args(self, resource_info, client_resource_info):
        server_data = self.data
        model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        kw = self.server_address
        kw.update({'resource_info': resource_info})
        return server_data, model, kw

    def _get_client_args(self, client_id, resource_info):
        client_data = self.data
        kw = self.client_address
        kw['server_host'] = self.server_address['host']
        kw['server_port'] = self.server_address['port']
        kw['resource_info'] = resource_info
        return client_data, kw

    def run(self):
        if self.cfg.distribute.role == 'server':
            self.server.run()
            return self.server.best_results
        elif self.cfg.distribute.role == 'client':
            self.client.join_in()
            self.client.run()


# TODO: remove FedRunner (keep now for forward compatibility)
class FedRunner(object):
    """
    This class is used to construct an FL course, which includes `_set_up`
    and `run`.

    Arguments:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Warnings:
        ``FedRunner`` will be removed in the future, consider \
        using ``StandaloneRunner`` or ``DistributedRunner`` instead!
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        logger.warning('`federate.core.fed_runner.FedRunner` will be '
                       'removed in the future, please use'
                       '`federate.core.fed_runner.get_runner` to get '
                       'Runner.')
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using FedRunner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs

        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        self.unseen_clients_id = []
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check the completeness of msg_handler.
        self.check()

    def setup(self):
        if self.mode == 'standalone':
            self.shared_comm_queue = deque()
            self._setup_for_standalone()
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            trainer_representative = self.client[1].trainer
            if trainer_representative is not None:
                trainer_representative.print_trainer_meta_info()
        elif self.mode == 'distributed':
            self._setup_for_distributed()

    def _setup_for_standalone(self):
        """
        To set up server and client for standalone mode.
        """
        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num+1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()

        # assume the client-wise data are consistent in their input&output
        # shape
        self._shared_client_model = get_model(
            self.cfg, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None

        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

    def _setup_for_distributed(self):
        """
        To set up server or client for distributed mode.
        """

        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def run(self):
        """
        To run an FL course, which is called after server/client has been
        set up.
        For the standalone mode, a shared message queue will be set up to
        simulate ``receiving message``.
        """
        self.setup()
        if self.mode == 'standalone':
            # trigger the FL course
            for each_client in self.client:
                self.client[each_client].join_in()

            if self.cfg.federate.online_aggr:
                # any broadcast operation would be executed client-by-client
                # to avoid the existence of #clients messages at the same time.
                # currently, only consider centralized topology
                self._run_simulation_online()

            else:
                self._run_simulation()

            self.server._monitor.finish_fed_runner(fl_mode=self.mode)

            return self.server.best_results

        elif self.mode == 'distributed':
            if self.cfg.distribute.role == 'server':
                self.server.run()
                return self.server.best_results
            elif self.cfg.distribute.role == 'client':
                self.client.join_in()
                self.client.run()

    def _run_simulation_online(self):
        def is_broadcast(msg):
            return len(msg.receiver) >= 1 and msg.sender == 0

        cached_bc_msgs = []
        cur_idx = 0
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if is_broadcast(msg):
                    cached_bc_msgs.append(msg)
                    # assume there is at least one client
                    msg = cached_bc_msgs[0]
                    self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                    cur_idx += 1
                    if cur_idx >= len(msg.receiver):
                        del cached_bc_msgs[0]
                        cur_idx = 0
                else:
                    self._handle_msg(msg)
            elif len(cached_bc_msgs) > 0:
                msg = cached_bc_msgs[0]
                self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                cur_idx += 1
                if cur_idx >= len(msg.receiver):
                    del cached_bc_msgs[0]
                    cur_idx = 0
            else:
                # finished
                break

    def _run_simulation(self):
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if msg.receiver == [self.server_id]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up the server
        """
        self.server_id = 0
        if self.mode == 'standalone':
            if self.server_id in self.data:
                server_data = self.data[self.server_id]
                model = get_model(self.cfg,
                                  server_data,
                                  backend=self.cfg.backend)
            else:
                server_data = None
                data_representative = self.data[1]
                model = get_model(
                    self.cfg, data_representative, backend=self.cfg.backend
                )  # get the model according to client's data if the server
                # does not own data
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info,
                'client_resource_info': client_resource_info
            }
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
            kw = self.server_address
            kw.update({'resource_info': resource_info})
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.server_class:
            self._server_device = self.gpu_manager.auto_choice()
            server = self.server_class(
                ID=self.server_id,
                config=self.cfg,
                data=server_data,
                model=model,
                client_num=self.cfg.federate.client_num,
                total_round_num=self.cfg.federate.total_round_num,
                device=self._server_device,
                unseen_clients_id=self.unseen_clients_id,
                **kw)

            if self.cfg.nbafl.use:
                from federatedscope.core.trainers.trainer_nbafl import \
                    wrap_nbafl_server
                wrap_nbafl_server(server)

        else:
            raise ValueError

        logger.info('Server has been set up ... ')

        return server

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up the client
        """
        self.server_id = 0
        if self.mode == 'standalone':
            client_data = self.data[client_id]
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info
            }
        elif self.mode == 'distributed':
            client_data = self.data
            kw = self.client_address
            kw['server_host'] = self.server_address['host']
            kw['server_port'] = self.server_address['port']
            kw['resource_info'] = resource_info
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.client_class:
            client_specific_config = self.cfg.clone()
            if self.client_cfgs and \
                    self.client_cfgs.get('client_{}'.format(client_id)):
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    self.client_cfgs.get('client_{}'.format(client_id)))
                client_specific_config.freeze()
            client_device = self._server_device if \
                self.cfg.federate.share_local_model else \
                self.gpu_manager.auto_choice()
            client = self.client_class(ID=client_id,
                                       server_id=self.server_id,
                                       config=client_specific_config,
                                       data=client_data,
                                       model=client_model
                                       or get_model(client_specific_config,
                                                    client_data,
                                                    backend=self.cfg.backend),
                                       device=client_device,
                                       is_unseen_client=client_id
                                       in self.unseen_clients_id,
                                       **kw)
        else:
            raise ValueError

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return client

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the
        standalone mode)
        """
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes)

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return
