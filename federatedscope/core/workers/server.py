import logging
import copy
import os
import sys

import numpy as np
import pickle

from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict, add_prefix_to_path, get_ds_rank
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.workers.base_server import BaseServer

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)


class Server(BaseServer):
    """
    The Server class, which describes the behaviors of server in an FL \
    course. The behaviors are described by the handled functions (named as \
    ``callback_funcs_for_xxx``).

    Arguments:
        ID: The unique ID of the server, which is set to 0 by default
        state: The training round
        config: the configuration
        data: The data owned by the server (for global evaluation)
        model: The model used for aggregation
        client_num: The (expected) client num to start the FL course
        total_round_num: The total number of the training round
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        aggregators: a protocol for aggregate all clients' model(s), see \
            ``federatedscope.core.aggregators``
        sample_client_num: number of client aggregated in each round
        msg_buffer: dict buffer for storing message
        staled_msg_buffer: list buffer for storing staled message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):
        # 调用基类 BaseServer 的初始化方法，保存 ID、状态、配置和模型对象
        super(Server, self).__init__(ID, state, config, model, strategy)
        # Register message handlers # 注册默认的消息处理函数 (Handlers)
        # 这会将消息类型（如 'model_para'）映射到对应的处理方法（如 callback_funcs_model_para）
        self._register_default_handlers()

        # 如果没有配置，直接返回（通常用于未配置的 Worker 占位）
        if config is None:
            return

        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.early_stopper = EarlyStopper( # 根据配置初始化早停逻辑
            # 根据配置中的 patience (耐心值) 和 delta (阈值) 来决定是否提前结束训练
            self._cfg.early_stop.patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)
        # 如果是单机模式且开启共享模型，将模型放到指定设备(GPU/CPU)
        if self._cfg.federate.share_local_model \
                and not self._cfg.federate.process_num > 1 \
                and not self._cfg.llm.deepspeed.use \
                and not self._cfg.llm.accelerator.use:
            # if self._cfg.train.is_enable_half:
            #     model.to(torch.bfloat16)
            # put the model to the specified device
            model.to(device)
        # Build aggregator # 构建聚合器 (Aggregator)
        self.aggregator = get_aggregator(self._cfg.federate.method,
                                         model=model,
                                         device=device,
                                         online=self._cfg.federate.online_aggr,
                                         config=self._cfg)
        # 从检查点恢复模型 (Restore)，如果配置了 restore_from，则加载之前的模型权重和训练轮次
        if self._cfg.federate.restore_from != '':
            if not os.path.exists(self._cfg.federate.restore_from):
                logger.warning(f'Invalid `restore_from`:'
                               f' {self._cfg.federate.restore_from}.')
            else:
                comm_round = \
                    self.aggregator.load_model(self._cfg.federate.restore_from) + 1
                logger.info(f"Restored the model from "
                            f"{self._cfg.federate.restore_from}")
                # See if we resume from the last checkpoint
                _, save_fname = os.path.split(self._cfg.federate.save_to)
                if save_fname in self._cfg.federate.restore_from:
                    self.state = comm_round
                    logger.info(
                        "The training resumes from the last checkpoint "
                        f"at {comm_round}-th round. ")
        # 检查每个 Trainer 维护的模型数量配置是否合法
        if int(config.model.model_num_per_trainer) != \
                config.model.model_num_per_trainer or \
                config.model.model_num_per_trainer < 1:
            raise ValueError(
                f"model_num_per_trainer should be integer and >= 1, "
                f"got {config.model.model_num_per_trainer}.")
        self.model_num = config.model.model_num_per_trainer
        self.models = [self.model]
        self.aggregators = [self.aggregator]
        # 处理 model_num_per_trainer > 1 的情况 (例如集成学习)，需要维护多个模型，因此复制多份模型和聚合器
        if self.model_num > 1: 
            self.models.extend(
                [copy.deepcopy(self.model) for _ in range(self.model_num - 1)])
            self.aggregators.extend([
                copy.deepcopy(self.aggregator)
                for _ in range(self.model_num - 1)
            ])

        # function for recovering shared secret  # 初始化秘密共享恢复函数，用于安全聚合的解密阶段
        self.recover_fun = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num)
        ).fixedpoint2float if self._cfg.federate.use_ss else None

        # 如果开启全局评估，Server需要自己持有一个 Trainer
        if self._cfg.federate.make_global_eval:
            # set up a trainer for conducting evaluation in server
            assert self.models is not None
            assert self.data is not None
            # 创建仅用于评估的 Trainer
            self.trainer = get_trainer(
                model=self.models[0],
                data=self.data,
                device=self.device,
                config=self._cfg,
                only_for_eval=True,
                monitor=self._monitor
            )  # the trainer is only used for global evaluation
            self.trainers = [self.trainer]
            # 同样支持多模型评估
            if self.model_num > 1:
                # By default, the evaluation is conducted by calling
                # trainer[i].eval over all internal models
                self.trainers.extend([
                    copy.deepcopy(self.trainer)
                    for _ in range(self.model_num - 1)
                ])

        # Initialize the number of joined-in clients  # 客户端数量管理
        self._client_num = client_num
        self._total_round_num = total_round_num
        self.sample_client_num = int(self._cfg.federate.sample_client_num)
        self.join_in_client_num = 0 # 已加入的客户端计数
        self.join_in_info = dict()
        # the unseen clients indicate the ones that do not contribute to FL
        # process by training on their local data and uploading their local
        # model update. The splitting is useful to check participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        # 未见客户端 (Unseen Clients)：用于测试泛化能力的保留客户端，不参与训练
        self.unseen_clients_id = [] if unseen_clients_id is None \
            else unseen_clients_id

        # Server state
        self.is_finish = False  # 训练结束标志

        # Sampler # 初始化采样器，如果是 'uniform' (均匀采样)，直接初始化
        if self._cfg.federate.sampler in ['uniform']:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=None)
            # Initialize the sampler when loading from last checkpoint
            # 如果是从断点恢复 (state > 0)，需要让随机数生成器"空转"对应的轮次，
            # 以保证随机采样的序列与之前的实验一致（可复现性）
            if self.sample_client_num > 0:
                for _ in range(self.state): # 这里
                    temp = self.sampler.sample(size=self.sample_client_num)
                    self.sampler.change_state(temp, 'idle')
        else:
            # Some type of sampler would be instantiated in trigger_for_start,
            # since they need more information
            # 其他类型的采样器可能需要更多信息，延迟到 trigger_for_start 时初始化
            self.sampler = None

        # Current Timestamp # 初始化时间戳，用于异步 FL 的超时控制
        self.cur_timestamp = 0
        self.deadline_for_cur_round = 1

        # Staleness toleration # 过时容忍度 (Staleness Toleration)：允许接收多少轮之前的旧模型
        self.staleness_toleration = self._cfg.asyn.staleness_toleration if \
            self._cfg.asyn.use else 0
        self.dropout_num = 0  # 掉线/丢弃的客户端数量

        # Device information # 保存硬件资源信息（用于模拟异构环境）
        self.resource_info = kwargs['resource_info'] \
            if 'resource_info' in kwargs else None
        self.client_resource_info = kwargs['client_resource_info'] \
            if 'client_resource_info' in kwargs else None

        # Initialize communication manager and message buffer # 初始化消息缓冲区
        self.msg_buffer = {'train': dict(), 'eval': dict()} # 存储当轮消息
        self.staled_msg_buffer = list() # 存储过时消息
        # 根据运行模式初始化通信管理器
        if self.mode == 'standalone':
            comm_queue = kwargs.get('shared_comm_queue', None)
            # 单机模式：使用队列 (Queue) 或 DDP
            if self._cfg.federate.process_num > 1:
                # 如果是多进程模拟 (DDP)，使用 StandaloneDDPCommManager
                id2comm = kwargs.get('id2comm', None)
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue,
                    monitor=self._monitor,
                    id2comm=id2comm)
            else:
                # 普通单机模拟，使用基于内存队列的 StandaloneCommManager
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
        elif self.mode == 'distributed':
            # --- 真实分布式模式 ---
            # 使用 gRPC 通信管理器监听端口
            host = kwargs['host']
            port = kwargs['port']
            self.comm_manager = gRPCCommManager(host=host,
                                                port=port,
                                                client_num=client_num,
                                                cfg=self._cfg.distribute)
            logger.info('Server: Listen to {}:{}...'.format(host, port))

        # inject noise before broadcast # 差分隐私噪声注入器 (Placeholder)
        self._noise_injector = None

    @property
    def client_num(self):
        return self._client_num

    @client_num.setter
    def client_num(self, value):
        self._client_num = value

    @property
    def total_round_num(self):
        return self._total_round_num

    @total_round_num.setter
    def total_round_num(self, value):
        self._total_round_num = value

    def register_noise_injector(self, func):
        self._noise_injector = func

    # 这是 Server 的主入口函数。
    def run(self):
        """
        To start the FL course, listen and handle messages (for distributed \
        mode).
        """

        # Begin: Broadcast model parameters and start to FL train 
        # 1. 第一阶段：等待客户端加入 (Client Join-in Phase)
        # 循环检查已加入的客户端数量是否达到了预设的 client_num
        while self.join_in_client_num < self.client_num:
            # 阻塞式接收消息（此时主要接收 'join_in' 类型的消息）
            msg = self.comm_manager.receive()
            # 根据消息类型调用对应的处理函数
            # 如果是 'join_in'，会增加 self.join_in_client_num 计数
            # 当数量达标时，会在 handler 内部触发 trigger_for_start()
            self.msg_handlers[msg.msg_type](msg)  # 处理 Join 消息
        # 这是一个阻塞循环。在分布式设置中，Server 启动后会在这里挂起，直到配置中指定的 client_num 个客户端都发送了握手消息。
        # 一旦人齐了，最后一条消息的处理函数会触发 trigger_for_start，广播初始模型，从而正式进入下一阶段。

        # Running: listen for message (updates from clients),
        # aggregate and broadcast feedbacks (aggregated model parameters)
        # 2. 第二阶段：主循环配置准备 在进入训练循环前，设置异步/同步相关的参数和超时控制。
        # 确定每一轮聚合所需的最小接收消息数
        # 如果开启了异步训练 (cfg.asyn.use)，使用异步配置的最小值；
        # 否则（同步模式），使用采样客户端数量 (sample_client_num)
        min_received_num = self._cfg.asyn.min_received_num \
            if self._cfg.asyn.use else self._cfg.federate.sample_client_num
        # 初始化失败计数器，用于记录连续超时的次数
        num_failure = 0
        # 获取时间预算（Time Budget），用于异步训练的超时控制
        # 如果没开启异步，默认是 -1（代表无限等待，不超时）
        time_budget = self._cfg.asyn.time_budget if self._cfg.asyn.use else -1
        # 使用 Timeout 上下文管理器来监控代码块的执行时间
        # 这里定义了“什么情况下算这一轮结束”。同步模式下通常需要收到所有被采样客户端的回复；
        # 异步模式下则可能只需要收到一部分。Timeout 上下文管理器用于防止 Server 因为某些 Client 掉线而永久死锁
        # 3. 第三阶段：联邦学习主循环 (Main FL Loop)
        with Timeout(time_budget) as time_counter:
            # 只要当前轮次 (self.state) 小于等于总轮次 (total_round_num)，就继续循环
            while self.state <= self.total_round_num:
                try:
                    # --- 正常消息处理流程 ---
                    # 1. 阻塞接收消息（通常是 'model_para' 或 'metrics'）
                    msg = self.comm_manager.receive() # Server 在此等待客户端发回梯度或评估结果
                    # 2. 调用对应的 handler 处理消息
                    # move_on_flag 是一个布尔值，表示是否完成了当前轮次的聚合任务并进入了下一轮
                    # 如果 handler 内部触发了聚合 (check_and_move_on 返回 True)，这里就是 True
                    move_on_flag = self.msg_handlers[msg.msg_type](msg)  # 会处理具体的业务逻辑（如缓存梯度、计算聚合）
                    # 3. 如果成功推进了轮次（move_on_flag 为 True），重置超时计时器
                    if move_on_flag:
                        time_counter.reset()
                # 4. 第四阶段：异常处理与超时恢复 (Exception Handling & Recovery)
                except TimeoutError:
                    # 记录超时日志
                    logger.info('Time out at the training round #{}'.format(
                        self.state))
                    # --- 尝试强制推进 ---
                    # 即使超时了，先检查一下缓冲区里的消息是否勉强够用了？
                    # 检查评估结果缓冲区
                    move_on_flag_eval = self.check_and_move_on(
                        min_received_num=min_received_num,
                        check_eval_result=True)
                    # 检查训练结果缓冲区
                    move_on_flag = self.check_and_move_on(
                        min_received_num=min_received_num)
                    # --- 确实无法推进，判定为失败 ---
                    if not move_on_flag and not move_on_flag_eval:
                        num_failure += 1
                        # Terminate the training if the number of failure
                        # exceeds the maximum number (default value: 10)
                        # 如果连续失败次数超过阈值（默认通常是 10 次），则彻底终止训练
                        if time_counter.exceed_max_failure(num_failure):
                            logger.info(f'----------- Training fails at round '
                                        f'#{self.state}-------------')
                            break

                        # Time out, broadcast the model para and re-start
                        # the training round
                        # --- 重启当前轮次 (Soft Restart) ---
                        # 如果还没达到最大失败次数，尝试重发模型来"唤醒"客户端
                        logger.info(
                            f'----------- Re-starting the training round ('
                            f'Round #{self.state}) for {num_failure} time '
                            f'-------------')
                        # TODO: Clean the msg_buffer
                        # 清理当前轮次的垃圾数据（旧消息）
                        if self.state in self.msg_buffer['train']:
                            self.msg_buffer['train'][self.state].clear()

                        # 【关键操作】重新广播当前的模型参数
                        # 这相当于告诉客户端："这一轮重来，请重新训练/发送"
                        self.broadcast_model_para(
                            msg_type='model_para',
                            sample_client_num=self.sample_client_num)
                    else:
                    # 如果通过 check_and_move_on 成功推进了，说明超时是误报或勉强通过，重置失败计数 num_failure = 0
                        num_failure = 0
                    # 无论结果如何，重置计时器，开始新的等待周期
                    time_counter.reset()
        # 发送结束信号给所有客户端，并关闭 Monitor 等资源
        self.terminate(msg_type='finish')
        # 在真实网络环境中，丢包或节点掉线很常见。Server不会立即崩溃，而是尝试检查现有数据是否足够进行“有损”聚合,
        # 如果不够，尝试重新广播模型（Re-start Round），给客户端第二次机会。只有连续多次失败，才会彻底放弃。

    # 检查缓冲区中的消息是否已满足条件，同时如果满足条件则执行操作（客户端当前轮数全部完成执行聚合，整体完成执行评估）
    def check_and_move_on(self,
                          check_eval_result=False, # 检查的是评估结果（metrics）还是训练结果（梯度/模型参数）
                          min_received_num=None): # 手动指定最小接收数量，通常用于异步模式的超时控制逻辑中
        """
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """
        # 收到多少个客户端的回复才算这一轮结束
        if min_received_num is None:
            if self._cfg.asyn.use:
                # 异步模式：使用配置中指定的最小接收数 (min_received_num)
                min_received_num = self._cfg.asyn.min_received_num
            else:
                # 同步模式：必须收到所有被采样客户端 (sample_client_num) 的回复
                min_received_num = self._cfg.federate.sample_client_num
        # 安全检查：接收阈值不能超过总采样人数
        assert min_received_num <= self.sample_client_num

        # 特殊情况：单机模拟模式下的评估阶段
        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            # 在单机模拟评估时，通常强制要求收到所有客户端的结果（强同步）
            # min_received_num 被重置为当前通信管理器中邻居的总数（即所有客户端）
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training # 预设为 True，如果 check_buffer 失败则改为 False
        # round or finishing the evaluation
        # 调用 check_buffer 方法检查当前轮次 (self.state) 的消息数量是否 >= min_received_num
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            # 处理训练消息，说明收集到了足够的模型更新，需要进行聚合
            if not check_eval_result:
                # Receiving enough feedback in the training process
                # 调用聚合器将收集到的梯度/参数进行加权平均，更新全局模型，aggregated_num 返回实际参与聚合的客户端数量
                aggregated_num = self._perform_federated_aggregation()
                self.state += 1
                # 定期评估：如果当前轮次满足评估频率要求，且还没结束
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    # 触发评估流程
                    if not self._cfg.llm.adapter.moe:
                        self.eval()

                # 开启下一轮训练
                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    # 清理上一轮 (state-1) 的训练消息缓冲区，释放内存
                    self.msg_buffer['train'][self.state - 1].clear()
                    # 初始化新一轮的消息字典
                    self.msg_buffer['train'][self.state] = dict()
                    # 清理过时消息缓冲区
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    # 广播新的模型参数，开始新一轮训练
                    self._start_new_training_round(aggregated_num)
                else: # 训练结束，最终评估
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    if not self._cfg.llm.adapter.moe:
                        self.eval()

            else: # 收集到了足够的评估指标（如 Accuracy, Loss）
                # Receiving enough feedback in the evaluation process
                # 将所有客户端发来的评估指标合并、计算平均值、记录最佳结果
                self._merge_and_format_eval_results()
                # 如果轮次已达上限，将 Server 标记为结束状态
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else: # 如果 check_buffer 返回 False（消息还不够），将 flag 设为 False
            move_on_flag = False

        return move_on_flag

    # 评估结束后被调用，负责早停检查、模型保存以及训练结束后的收尾工作
    def check_and_save(self): 
        """
        To save the results and save model after each evaluation, and check \
        whether to early stop.
        """

        # early stopping
        # --- 检查历史结果中的指标 ---
        # 优先检查加权平均结果 (Results_weighted_avg)
        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_weighted_avg']:
            # 调用 early_stopper 跟踪指标变化，返回是否应该停止
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        # 其次检查平均结果 (Results_avg)
        elif "Results_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        else:
            # 如果找不到对应的指标，默认不停止
            should_stop = False

        if should_stop:
            # 通知 Monitor 记录收敛状态
            self._monitor.global_converged()
            # 发送 "converged" 消息给所有邻居 (客户端)
            # 这通常会让客户端跳出它们的训练循环
            self.comm_manager.send(
                Message(
                    msg_type="converged",
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    timestamp=self.cur_timestamp,
                    state=self.state,
                ))
            # 【关键】强制修改当前状态为 "总轮次 + 1"
            # 这确保了 Server 主循环 (run 方法中的 while 循环) 会在下一次检查时退出
            self.state = self.total_round_num + 1

        # 定期保存模型
        if self.state != self.total_round_num and \
                self.state % self._cfg.federate.save_freq == 0 and \
                self._cfg.federate.save_freq > 0:
            # 构造保存路径，例如 "100_model.pth"
            path = add_prefix_to_path(f'{self.state}_',
                                      self._cfg.federate.save_to)
            # 只有主进程 (rank 0) 执行保存操作，避免多进程写入冲突

            # if self.ds_rank == 0:
            #     self.aggregator.save_model(path, self.state)
            logger.info("The model has been saved !")
            self.aggregator.save_model(path, self.state)

        # 条件：触发早停 或 已经到达最后一轮
        if should_stop or self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round or early stopped
            # 1. 保存最佳结果 (Best Results)
            self.save_best_results()
            # 2. 保存客户端的评估结果
            # 如果不是在 Server 端做全局评估 (即是在 Client 端做的评估)，则保存所有 Client 的详细日志
            if not self._cfg.federate.make_global_eval:
                self.save_client_eval_results()
            # 3. 发送结束信号 (terminate)
            # 这会广播 "finish" 消息给所有客户端，彻底关闭整个联邦学习任务
            self.terminate(msg_type='finish')

        # Clean the clients evaluation msg buffer
        # 如果评估是在 Client 端进行的 (Server 端收到了评估消息)
        if not self._cfg.federate.make_global_eval:
            # 获取当前评估轮次的 Key
            round = max(self.msg_buffer['eval'].keys())
            # 清空该轮次的评估消息缓冲区
            self.msg_buffer['eval'][round].clear()

        # 这是一个针对 run() 方法中 while 循环的特殊处理
        # 如果当前刚好是最后一轮 (total_round_num)
        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            # 强制 +1，使得 state > total_round_num
            # 这样 run() 方法里的 while self.state <= self.total_round_num 就会在下一次判断时为 False，从而退出循环
            self.state += 1

    # 收集缓冲区中的所有客户端更新（包括新鲜的和过时的），提取模型参数，计算监控指标，并调用聚合算法更新全局模型
    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        # 获取当前轮次的所有训练消息 (字典结构: ClientID -> MessageContent)
        train_msg_buffer = self.msg_buffer['train'][self.state]
        # 循环处理每一个模型
        for model_idx in range(self.model_num):
            model = self.models[model_idx] # 获取当前要处理的模型对象
            aggregator = self.aggregators[model_idx] # 获取对应的聚合器实例
            msg_list = list()  # 用于存储待聚合的参数列表
            staleness = list() # 用于存储每个更新的过时程度 (Staleness)

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1: # --- 情况 A：单模型 ---
                    msg_list.append(train_msg_buffer[client_id]) # 直接添加消息内容 (通常是 (sample_size, model_para))
                else: # --- 情况 B：多模型 ---
                    # 消息内容包含多个模型的参数列表，需提取对应 model_idx 的那个
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id] 
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                # 记录过时度：因为是当前轮次的消息，过时度为 0
                staleness.append((client_id, 0))

            # 遍历过时消息缓冲区 (self.staled_msg_buffer)
            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message  # 解包：消息轮次、发送者、内容
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # 计算过时度：当前轮次 - 消息轮次
                # 例如当前是第10轮，收到第8轮的消息，staleness = 2
                staleness.append((client_id, self.state - state))

            # Trigger the monitor here (for training)
            # Monitor计算这些更新的统计特性（例如梯度范数、更新方差等），用于系统监控或自适应算法
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            # 统计参与聚合的消息总数
            aggregated_num = len(msg_list)
            # 构造聚合信息包
            agg_info = {
                'client_feedback': msg_list, # 包含 (样本数, 参数) 的列表
                'recover_fun': self.recover_fun, # 用于安全聚合的恢复函数 (如果开启了 Secret Sharing)
                'staleness': staleness,  # 过时度列表 (用于异步加权)
            }
            # logger.info(f'The staleness is {staleness}')
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            # 使用 merge_param_dict 合并参数
            # 这里的逻辑是为了处理"懒加载" (Lazy Load) 或部分更新的情况
            # copy() 确保不修改原始状态字典
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            # 将合并后的参数加载回 PyTorch 模型
            # strict=False 允许部分键值不匹配（增加了鲁棒性）
            model.load_state_dict(merged_param, strict=False)

        # 返回参与聚合的客户端数量，供后续逻辑判断（如异步训练是否需要补偿广播）
        return aggregated_num

    # 启动一个新的训练轮次，同步模式下，它负责采样新一批客户端并广播模型，异步模式下，它负责更新时间窗口截止线，并根据刚刚完成聚合的数量“补货”新的客户端
    def _start_new_training_round(self, aggregated_num=0):
        """
        The behaviors for starting a new training round
        """
        if self._cfg.asyn.use:  # for asynchronous training
            # --- 1. 更新时间窗口 (Time Budget) ---
            # 如果聚合策略是基于时间限制的 ("time_up")
            if self._cfg.asyn.aggregator == "time_up":
                # Update the deadline according to the time budget
                # 计算本轮的截止时间 = 当前时间戳 + 时间预算
                # 这意味着 Server 会在 deadline 之前持续收集消息，超时后才进行强制聚合
                self.deadline_for_cur_round = \
                    self.cur_timestamp + self._cfg.asyn.time_budget
            # --- 2. 广播策略 (Broadcast Manner) ---
            # 如果配置为 "after_aggregating"：意味着每完成一次聚合，就广播一次新模型
            # (与之对应的是 "after_receiving"，即收到一个就发一个，通常在 callback 中处理)
            if self._cfg.asyn.broadcast_manner == \
                    'after_aggregating':
                # --- 3. 确定补货数量 (Replenishment Strategy) ---
                # 场景 A: 开启了超额选择 (Overselection)
                # 无论之前完成了多少，都重新采样固定数量 (sample_client_num) 的客户端
                if self._cfg.asyn.overselection:
                    sample_client_num = self.sample_client_num
                # 场景 B: 维持并发度 (Concurrency Maintenance)
                # 这是一个"完成多少，补多少"的逻辑。
                # 补货数 = 刚刚完成聚合的客户端数 (aggregated_num) + 掉线/丢弃的客户端数 (dropout_num)
                # 这保证了系统中正在训练的客户端总数维持在一个稳定水平
                else:
                    sample_client_num = aggregated_num + \
                                        self.dropout_num

                # --- 4. 广播模型 ---
                # 调用广播函数，将最新的全局模型发送给新选出的客户端
                self.broadcast_model_para(msg_type='model_para',
                                          sample_client_num=sample_client_num)
                # 重置掉线计数器
                self.dropout_num = 0
        else:  # for synchronous training
            # 在同步模式下，每一轮开始的操作非常简单直接：
            # 1. 采样 sample_client_num 个客户端
            # 2. 广播最新的全局模型给它们
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    # 服务器收到足够数量的客户端评估反馈后被调用，其核心职责是汇总所有客户端的指标、更新历史记录，并触发后续的保存或早停检查逻辑
    def _merge_and_format_eval_results(self):
        """
        The behaviors of server when receiving enough evaluating results
        """
        # Get all the message & aggregate
        # 调用辅助函数 merge_eval_results_from_all_clients()
        # 该函数会遍历 msg_buffer['eval'] 中所有客户端的数据，区分"参与训练的客户端"和"未见客户端"，
        # 计算平均值（如 Avg Loss, Avg Accuracy）并进行格式化。
        formatted_eval_res = \
            self.merge_eval_results_from_all_clients()
        # 将本轮刚刚汇总好的结果 (formatted_eval_res) 合并到服务器的总历史记录 (self.history_results) 中
        # merge_dict_of_results 是一个工具函数，负责将新字典的内容追加到旧字典中
        self.history_results = merge_dict_of_results(self.history_results,
                                                     formatted_eval_res)
        # WandB 系统指标监控 (仅单机模拟模式) 检查三个条件：
        # 1. 运行模式是 'standalone' (单机模拟)
        # 2. 开启了 WandB 在线追踪
        # 3. 确实使用了 WandB
        if self.mode == 'standalone' and \
                self._monitor.wandb_online_track and \
                self._monitor.use_wandb:
            # 在单机模拟中，系统指标（如CPU/内存）通常是集中记录的
            # 这里手动触发 Monitor 合并这些模拟模式下的系统指标
            self._monitor.merge_system_metrics_simulation_mode(
                file_io=False, from_global_monitors=True)
        # 评估流程的最后一步
        self.check_and_save()
    
    # 保存最终的模型文件以及持久化记录整个训练过程中的最佳评估指标
    def save_best_results(self):
        """
        To Save the best evaluation results.
        """
        # Save final round model
        # 一. 保存最终模型
        # 检查两个条件：
        # 1. 配置中指定了保存路径 (save_to != '')
        # 2. 当前进程是主进程 (ds_rank == 0)。防止多进程环境下多个进程同时写入同一个文件导致损坏。
        if self._cfg.federate.save_to != '' and self.ds_rank == 0:
            # 调用聚合器来执行实际的保存操作
            # add_prefix_to_path: 辅助函数，给文件名加上前缀 'final_'，区分中间的 checkpoint
            # self.state: 当前的训练轮次
            self.aggregator.save_model(
                add_prefix_to_path('final_', self._cfg.federate.save_to),
                self.state)
        # 二. 格式化最佳结果 
        # 调用 Monitor 组件将 best_results 字典转换为便于记录或打印的格式
        formatted_best_res = self._monitor.format_eval_res(
            results=self.best_results, # 传入最佳结果数据
            rnd="Final",  # 标记轮次为 "Final"
            role='Server #', # 标记角色
            forms=["raw"], # 请求原始格式
            return_raw=True) # 返回原始数据对象
        # 将格式化后的最佳结果打印到控制台日志
        logger.info(formatted_best_res)
        # 将结果保存到文件（通常是 JSON 或文本文件，取决于 Monitor 的配置）
        # 这确保了实验结果被永久记录，便于后续分析
        self._monitor.save_formatted_results(formatted_best_res)

    # 在训练结束时（早停或完成），将每个客户端最后一次上传的本地评估结果保存到日志文件中
    def save_client_eval_results(self):
        """
        save the evaluation results of each client when the fl course \
        early stopped or terminated
        """
        # 获取评估消息缓冲区中最大的轮次 Key (即最新的一轮评估)
        rnd = max(self.msg_buffer['eval'].keys()) # self.msg_buffer['eval'] 是一个字典，结构为 {round: {client_id: metrics}}
        # 提取该轮次下所有客户端的评估数据
        eval_msg_buffer = self.msg_buffer['eval'][rnd]

        # 打开输出目录下的 "eval_results.log" 文件
        # 使用 "a" (append) 模式，表示追加写入，保留之前的内容
        with open(os.path.join(self._cfg.outdir, "eval_results.log"),
                  "a") as outfile:
            # 遍历缓冲区中每个客户端的数据
            # client_id: 客户端 ID
            # client_eval_results: 该客户端上传的具体指标字典 (如 {'loss': 0.5, 'acc': 0.8})
            for client_id, client_eval_results in eval_msg_buffer.items():
                # 调用 Monitor 组件对结果进行格式化
                # return_raw=True 表示返回原始数据结构，方便转为字符串存储
                formatted_res = self._monitor.format_eval_res(
                    client_eval_results,
                    rnd=self.state,
                    role='Client #{}'.format(client_id),
                    return_raw=True)
                # 将格式化后的结果打印到控制台日志
                logger.info(formatted_res)
                # 将结果转换为字符串并写入文件，每条记录换一行
                outfile.write(str(formatted_res) + "\n")

    def merge_eval_results_from_all_clients(self):
        """
        Merge evaluation results from all clients, update best, \
        log the merged results and save them into eval_results.log

        Returns:
            the formatted merged results
        """
        round = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][round]
        eval_res_participated_clients = []
        eval_res_unseen_clients = []
        for client_id in eval_msg_buffer:
            if eval_msg_buffer[client_id] is None:
                continue
            if client_id in self.unseen_clients_id:
                eval_res_unseen_clients.append(eval_msg_buffer[client_id])
            else:
                eval_res_participated_clients.append(
                    eval_msg_buffer[client_id])

        formatted_logs_all_set = dict()
        for merge_type, eval_res_set in [("participated",
                                          eval_res_participated_clients),
                                         ("unseen", eval_res_unseen_clients)]:
            if eval_res_set != []:
                metrics_all_clients = dict()
                for client_eval_results in eval_res_set:
                    for key in client_eval_results.keys():
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(
                            float(client_eval_results[key]))
                formatted_logs = self._monitor.format_eval_res(
                    metrics_all_clients,
                    rnd=round,
                    role='Server #',
                    forms=self._cfg.eval.report)
                if merge_type == "unseen":
                    for key, val in copy.deepcopy(formatted_logs).items():
                        if isinstance(val, dict):
                            # to avoid the overrides of results using the
                            # same name, we use new keys with postfix `unseen`:
                            # 'Results_weighted_avg' ->
                            # 'Results_weighted_avg_unseen'
                            formatted_logs[key + "_unseen"] = val
                            del formatted_logs[key]
                logger.info(formatted_logs)
                formatted_logs_all_set.update(formatted_logs)
                self._monitor.update_best_result(
                    self.best_results,
                    metrics_all_clients,
                    results_type="unseen_client_best_individual"
                    if merge_type == "unseen" else "client_best_individual")

                self._monitor.save_formatted_results(formatted_logs)

                update_prior = -1  # Bigger the higher priority
                update_prior_list = ['fairness', 'avg', 'weighted_avg']
                update_best_this_round = False
                for form in self._cfg.eval.report:
                    if form in update_prior_list:
                        update_prior_tmp = update_prior_list.index(form)
                    else:
                        update_prior_tmp = -1
                    if form != "raw":
                        metric_name = form + "_unseen" if merge_type == \
                                                          "unseen" else form
                        update_best_this_round_tmp = \
                            self._monitor.update_best_result(
                                self.best_results,
                                formatted_logs[f"Results_{metric_name}"],
                                results_type=f"unseen_client_summarized_{form}"
                                if merge_type == "unseen" else
                                f"client_summarized_{form}")
                        if update_prior_tmp >= update_prior:
                            update_prior = update_prior_tmp
                            update_best_this_round = update_best_this_round_tmp
                if update_best_this_round:
                    # When the frequency of evaluations is high,
                    # the frequency of writing to disk in the early stages
                    # may also be high
                    if self._cfg.federate.save_to != '' and self.ds_rank == 0:
                        self.aggregator.save_model(self._cfg.federate.save_to,
                                                   self.state)

        return formatted_logs_all_set

    # Server 向 Client 分发全局模型
    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast \
                behavior. And ``sample_client_num = -1`` denotes to \
                broadcast to all the clients.
            filter_unseen_clients: whether filter out the unseen clients that \
                do not contribute to FL process by training on their local \
                data and uploading their local model update. The splitting is \
                useful to check participation generalization gap in [ICLR'22, \
                What Do We Mean by Generalization in Federated Learning?] \
                You may want to set it to be False when in evaluation stage
        """
        # 1. 过滤未见客户端，为了测试联邦学习模型的泛化能力，通常会保留一部分客户端不参与训练
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            # 通知采样器将unseen_clients_id标记为'unseen'状态，这样采样器在随后的sample()调用中就会自动忽略这些客户端
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        # 2. 客户端采样
        # --- 分支 A：部分采样 (通常用于训练) ---
        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else: # --- 分支 B：广播所有人 (通常用于评估或特殊模式) ---
            # broadcast to all clients
            # 获取所有已连接的邻居列表
            receiver = list(self.comm_manager.neighbors.keys())
            # 如果是发送模型参数 (训练任务)，手动将这些接收者标记为 'working' 状态
            # 以便采样器知道它们正在忙碌
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        # 3. 差分隐私噪声注入，如果在发送前需要对模型加噪（例如为了满足差分隐私要求）
        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                # 获取样本数量信息用于噪声计算
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                # 对Server端模型参数原地添加噪声
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])
        # 4. 准备模型参数
        # 优化：如果是 Local (纯本地训练) 或 Global (集中式训练)，Server 不需要广播参数
        # 因为 Local 模式不交换参数，Global 模式数据都在 Server
        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        # --- 情况 A：单机模拟且开启共享模型 ---
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            if self.model_num > 1:
                # 深拷贝模型：因为在单机模拟中对象是共享的，必须拷贝一份副本发送
                # 否则 Client 修改参数会直接影响 Server 的原始对象
                model_para = [
                    {} if skip_broadcast else copy.deepcopy(model.state_dict())
                    for model in self.models
                ]
            else:
                model_para = {} if skip_broadcast else copy.deepcopy(
                    self.models[0].state_dict())
        else:
        # --- 情况 B：标准模式 (分布式或不共享内存) ---
            if self.model_num > 1:
                model_para = [{} if skip_broadcast else model.state_dict()
                              for model in self.models]
            else:
                model_para = {} if skip_broadcast else self.models[
                    0].state_dict()

        # quantization
        # 5. 模型量化/压缩，为了节省通信带宽，对模型参数进行压缩。
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits # 量化位数 (如 8bit)
            # 对模型参数进行对称均匀量化
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        # We define the evaluation happens at the end of an epoch
        # 6. 发送消息
        # 计算消息中的轮次 (state)
        # 如果是评估消息 (evaluate)，通常指的是评估"上一轮"训练后的模型，所以 state - 1
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver, # 采样的全部客户端
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        # 如果是在线聚合 (Online Aggregation)，广播后需要重置聚合器状态
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        # 恢复采样器状态：将之前标记为 'unseen' 的客户端恢复为 'seen'
        # 以免影响后续逻辑
        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def broadcast_client_address(self):
        """
        To broadcast the communication addresses of clients (used for \
        additive secret sharing)
        """

        self.comm_manager.send(
            Message(msg_type='address',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=self.comm_manager.get_neighbors()))

    # Server判断缓冲区是否收集到了足够消息
    def check_buffer(self,
                     cur_round, # 当前检查的目标轮次
                     min_received_num, # 最小需要接收到的消息数量阈值
                     check_eval_result=False): # 标记位，True 表示检查评估结果缓冲区，False 表示检查训练（模型参数）缓冲区
        """
        To check the message buffer

        Arguments:
            cur_round (int): The current round number
            min_received_num (int): The minimal number of the receiving \
                messages
            check_eval_result (bool): To check training results for \
                evaluation results

        Returns
            bool: Whether enough messages have been received or not
        """

        if check_eval_result:
            # 安全检查：如果缓冲区里没有任何评估数据的 key，直接返回 False
            if 'eval' not in self.msg_buffer.keys() or len(
                    self.msg_buffer['eval'].keys()) == 0:
                return False

            buffer = self.msg_buffer['eval']
            # 获取当前最新的评估轮次 (使用 max 是为了容错，确保取到最新的)
            cur_round = max(buffer.keys())
            cur_buffer = buffer[cur_round]
            # 判断收到的评估消息数量是否达到阈值
            return len(cur_buffer) >= min_received_num
        else:
            # 获取当前轮次的缓冲区
            if cur_round not in self.msg_buffer['train']:
                cur_buffer = dict()
            else:
                cur_buffer = self.msg_buffer['train'][cur_round]
            # --- 异步模式逻辑 (Asynchronous Mode) ---
            # 如果开启了异步训练，且聚合策略是基于时间限制 ('time_up')
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                # 检查是否超时 (当前时间 >= 截止时间) 且 缓冲区完全为空 (连过时消息都没有)
                if self.cur_timestamp >= self.deadline_for_cur_round and len(
                        cur_buffer) + len(self.staled_msg_buffer) == 0:
                    # When the time budget is run out but the server has not
                    # received any feedback
                    # 警告：时间到了但没收到任何更新
                    logger.warning(
                        f'The server has not received any feedback when the '
                        f'time budget has run out, therefore the server would '
                        f'wait for more {self._cfg.asyn.time_budget} seconds. '
                        f'Maybe you should carefully reset '
                        f'`cfg.asyn.time_budget` to a reasonable value.')
                    # 自动延长截止时间 (增加一个 time_budget)
                    self.deadline_for_cur_round += self._cfg.asyn.time_budget
                    # 如果广播策略是 'after_aggregating'，尝试重新广播给掉线的客户端 (dropout_num)
                    # 试图"唤醒"它们
                    if self._cfg.asyn.broadcast_manner == \
                            'after_aggregating' and self.dropout_num != 0:
                        self.broadcast_model_para(
                            msg_type='model_para',
                            sample_client_num=self.dropout_num)
                        self.dropout_num = 0
                # 在异步 'time_up' 模式下，是否通过检查只取决于时间是否到了
                return self.cur_timestamp >= self.deadline_for_cur_round
            # --- 同步模式逻辑 (Synchronous Mode) --
            else:
                # 标准逻辑：当前轮次消息 + 过时消息 (Staled) >= 最小接收阈值
                return len(cur_buffer)+len(self.staled_msg_buffer) >= \
                       min_received_num

    # 检查是否所有预期的客户端 (client_num) 都已经成功加入了
    def check_client_join_in(self):
        """
        To check whether all the clients have joined in the FL course.
        """
        # 检查配置项 cfg.federate.join_in_info 是否不为空
        # 该配置通常是一个列表，例如 ['num_sample', 'client_resource']
        # 如果不为空，说明 Server 需要先收集到这些信息才算"加入成功"
        if len(self._cfg.federate.join_in_info) != 0:
            # self.join_in_info 是一个字典，存储了已收集到的客户端信息
            # 只有当字典的大小（即已上报信息的客户端数量）等于总预期的客户端数量 (self.client_num) 时
            # 才返回 True
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num

    # 初始化采样器、计算资源、建立拓扑，并最终广播初始模型以开始第一轮训练
    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """
        # 检查是否所有预期的客户端 (client_num) 都已经成功加入了
        # 如果人还没齐，直接返回，不执行后续启动逻辑
        if self.check_client_join_in():
            # 如果配置启用了秘密共享 (Secret Sharing) 或 纵向联邦 (Vertical FL)
            if self._cfg.federate.use_ss or self._cfg.vertical.use:
                # Server 广播所有已连接客户端的 IP 和端口信息给每一个客户端
                # 这样客户端之间才能建立点对点 (P2P) 连接
                self.broadcast_client_address()

            # get sampler 计算客户端资源信息
            # --- 分支 A：使用客户端上报的实时信息 ---
            # 如果在握手阶段 (callback_funcs_for_join_in)，客户端上报了 'client_resource'
            if 'client_resource' in self._cfg.federate.join_in_info:
                client_resource = [
                    self.join_in_info[client_index]['client_resource']
                    for client_index in np.arange(1, self.client_num + 1)
                ]
            # --- 分支 B：使用模拟环境的预设信息 ---
            else:
                # 计算模型大小 (用于估算通信延迟)
                if self._cfg.backend == 'torch':
                    try:
                        # 使用 pickle 序列化模型来计算其字节大小 (KB)
                        model_size = sys.getsizeof(pickle.dumps(
                            self.models[0])) / 1024.0 * 8.
                    except Exception as error:
                        model_size = 1.0
                        logger.warning(f'Error {error} in calculate model '
                                       f'size.')
                else:
                    # TODO: calculate model size for TF Model
                    # TensorFlow 后端暂不支持自动计算
                    model_size = 1.0
                    logger.warning(f'The calculation of model size in backend:'
                                   f'{self._cfg.backend} is not provided.')
                # 结合模型大小和预设的宽带/算力，计算每个客户端的预估延迟
                # 公式：通信延迟 (模型大小/带宽) + 计算延迟
                client_resource = [
                    model_size / float(x['communication']) +
                    float(x['computation']) / 1000.
                    for x in self.client_resource_info
                ] if self.client_resource_info is not None else None

            # 如果采样器还未初始化
            if self.sampler is None:
                self.sampler = get_sampler(
                    # 使用刚刚计算好的 client_resource 初始化采样器
                    # 这允许使用如 'weighted_random' 等考虑设备性能的采样策略
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.client_num,
                    client_info=client_resource)

            # change the deadline if the asyn.aggregator is `time up`
            # 如果开启了基于时间限制 (time_up) 的异步聚合
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                # 设置第一轮的截止时间 = 当前时间 + 时间预算
                self.deadline_for_cur_round = self.cur_timestamp + \
                                               self._cfg.asyn.time_budget

            # start feature engineering
            # 启动特征工程流程（默认为空操作，直接调用传入的函数）
            # 这里的核心动作是调用 self.broadcast_model_para
            # 发送类型为 'model_para' 的消息，携带初始模型，广播给采样的客户端
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        """
        Interface for feature engineering, the default operation is none
        """
        trigger_train_func(**kwargs_for_trigger_train_func)

    def trigger_for_time_up(self, check_timestamp=None):
        """
        The handler for time up: modify the currency timestamp \
        and check the trigger condition
        """
        if self.is_finish:
            return False

        if check_timestamp is not None and \
                check_timestamp < self.deadline_for_cur_round:
            return False

        self.cur_timestamp = self.deadline_for_cur_round
        self.check_and_move_on()
        return True

    def terminate(self, msg_type='finish'):
        """
        To terminate the FL course
        """
        self.is_finish = True
        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.models[0].state_dict()

        self._monitor.finish_fl()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=model_para))

    # 训练完全（获部分轮）后进行评估，评估可以在服务器端集中进行，也可以分发给客户端进行
    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """
        
        # --- 分支 A：服务器端全局评估 (Global Evaluation) ---
        # 如果配置为 True，表示 Server 拥有测试数据，将在 Server 端直接评估全局模型
        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i] # 获取对应的 Trainer 实例
                # Preform evaluation in server
                metrics = {} # 初始化指标字典
                # 遍历配置中定义的数据切分 (如 'test', 'val')
                for split in self._cfg.eval.split:
                    # 调用 Trainer 的 evaluate 方法，在指定数据集上计算指标
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)
                # --- 结果处理与记录 ---
                # 使用 Monitor 格式化评估结果
                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)
                # 更新"历史最佳"结果
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_eval_res['Results_raw'],
                    results_type="server_global_eval")
                # 将本次结果合并到历史结果记录中
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                # 保存格式化后的结果到文件
                self._monitor.save_formatted_results(formatted_eval_res)
                logger.info(formatted_eval_res)
            # --- 检查早停与保存 ---
            # 评估完成后，检查是否满足早停条件或需要保存 Checkpoint
            self.check_and_save()
        else: # Server 不自己算，而是将模型广播给客户端，让客户端算完后回传结果
            # Preform evaluation in clients
            # 调用 broadcast_model_para 发送 'evaluate' 类型的消息
            # 这会触发客户端的 callback_funcs_for_evaluate 方法
            # filter_unseen_clients=False 表示评估时不过滤"未见"客户端，通常希望所有人都参与评估
            self.broadcast_model_para(msg_type='evaluate',
                                      filter_unseen_clients=False)

    # 每当 Server 收到一个客户端完成训练并上传的消息时，就会调用此函数
    # 解量化、缓存消息（处理异步/过时消息）、触发聚合检查以及在异步模式下的即时广播
    def callback_funcs_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.

        Arguments:
            message: The received message.
        """
        # 如果 Server 已经标记为结束 (is_finish=True)，则不再处理任何新消息
        # 直接返回 'finish' 信号
        if self.is_finish:
            return 'finish'

        round = message.state  # 消息轮次
        sender = message.sender # 发送该消息的客户端ID
        timestamp = message.timestamp # 消息的时间戳
        content = message.content # 消息内容 (通常是 (sample_size, model_para))
        # 通知采样器该客户端已变为空闲状态 ('idle')
        # 这意味着该客户端已经完成了本轮任务，可以在下一轮再次被采样
        self.sampler.change_state(sender, 'idle')

        # dequantization 解量化 # 检查是否使用了 'uniform' 均匀量化方法
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            # content 结构通常是 (样本数, 模型参数)
            # content[1] 是模型参数
            if isinstance(content[1], list):  # multiple model
                sample_size = content[0]
                quant_model = [
                    symmetric_uniform_dequantization(x) for x in content[1]
                ]
            else:
                sample_size = content[0]
                quant_model = symmetric_uniform_dequantization(content[1])
            content = (sample_size, quant_model)

        # update the currency timestamp according to the received message
        # 更新 Server 的当前逻辑时间戳
        # 这对于异步联邦学习 (Asynchronous FL) 的超时控制非常重要
        assert timestamp >= self.cur_timestamp  # for test # 确保时间是单调递增的
        self.cur_timestamp = timestamp

        # --- 分支 A：当前轮次消息 ---
        if round == self.state:
            # 如果缓冲区还没初始化该轮次，先创建字典
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            # 将消息存入当前轮次的训练缓冲区
            self.msg_buffer['train'][round][sender] = content
        # --- 分支 B：过时但可容忍的消息 (Staled) ---
        # 如果消息轮次落后于当前轮次，但在容忍阈值 (staleness_toleration) 之内
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content)) # 存入过时消息缓冲区 (list结构)，后续聚合时会一并考虑
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1 # 增加丢弃计数（可能用于后续的补偿策略）

        # 如果开启了在线聚合 (流式聚合)
        # 不等待收集齐所有数据，而是每收到一个就立即聚合到主模型中
        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        # 调用 check_and_move_on 检查缓冲区是否已经收集了足够的消息
        # 如果够了，该函数内部会触发聚合、更新轮次、广播新模型
        # move_on_flag 表示是否成功推进到了下一轮
        move_on_flag = self.check_and_move_on()
        # 如果是异步模式 且 广播策略是 'after_receiving' (收到更新即广播)
        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            # 立即广播最新的模型参数给 1 个客户端
            # 这样可以让空闲的客户端立即开始新的训练，最大化资源利用率
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag

    # 处理客户端的加入请求、分配客户端 ID、收集客户端元数据，最终触发训练流程
    def callback_funcs_for_join_in(self, message: Message):
        """
        The handling function for receiving the join in information. The \
        server might request for some information (such as \
        ``num_of_samples``) if necessary, assign IDs for the servers. \
        If all the clients have joined in, the training process will be \
        triggered.

        Arguments:
            message: The received message
        """
        # 当客户端响应了服务器的“信息查询请求”（例如服务器询问客户端有多少数据量）时，会进入此分支。
        # --- 分支 1：处理带有 'info' 的消息 (如 'join_in_info') ---
        # 这通常是客户端回复 Server 之前发出的 'ask_for_join_in_info' 请求
        if 'info' in message.msg_type:
            sender, info = message.sender, message.content
            # 验证：确保客户端发来的信息包含配置中要求的所有字段
            for key in self._cfg.federate.join_in_info:
                assert key in info
            self.join_in_info[sender] = info # 存储信息
            logger.info('Server: Client #{:d} has joined in !'.format(sender)) # 记录已加入并完成信息交互
        else:
        # 当客户端第一次发起连接（发送 join_in 消息）时，进入此分支。这主要用于建立通信链路和分配ID
        # # --- 分支 2：处理标准的 'join_in' 请求 ---
            self.join_in_client_num += 1
            sender, address = message.sender, message.content
            # --- 情况 A：动态分配 ID (通常用于分布式模式) ---
            # 如果客户端发送的 ID 是 -1，说明它还不知道自己是谁，请求 Server 分配 ID
            if int(sender) == -1:  # assign number to client
                sender = self.join_in_client_num
                # 在通信管理器中注册该邻居 (ID -> Address 映射)
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)
                # 发送 'assign_client_id' 消息回客户端，告诉它"你是 Client #N"
                self.comm_manager.send(
                    Message(msg_type='assign_client_id',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=str(sender)))
            # --- 情况 B：静态 ID (通常用于单机模拟模式) ---
            # 客户端已经知道自己的 ID，直接注册地址即可
            else:
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)

            # --- 额外信息收集 ---
            # 如果配置中要求收集客户端信息 (例如 cfg.federate.join_in_info=['num_sample'])
            # Server 会立即发送一个反向请求 'ask_for_join_in_info'
            if len(self._cfg.federate.join_in_info) != 0:
                self.comm_manager.send(
                    Message(msg_type='ask_for_join_in_info',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=self._cfg.federate.join_in_info.copy()))
        
        # 调用trigger_for_start方法，每次收到消息后，都会尝试启动训练
        # 该方法会检查 self.join_in_client_num == self.client_num
        # 如果所有预期中的客户端都已加入（且完成了信息收集），则正式广播模型开始训练
        self.trigger_for_start()

    def callback_funcs_for_metrics(self, message: Message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """

        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()

        self.msg_buffer['eval'][rnd][sender] = content

        return self.check_and_move_on(check_eval_result=True)

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
