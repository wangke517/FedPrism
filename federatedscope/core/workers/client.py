import copy
import logging
import sys
import pickle

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    calculate_time_cost, add_prefix_to_path, get_ds_rank
from federatedscope.core.workers.base_client import BaseClient

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)


class Client(BaseClient):
    """
    The Client class, which describes the behaviors of client in an FL \
    course. The behaviors are described by the handling functions (named as \
    ``callback_funcs_for_xxx``)

    Arguments:
        ID: The unique ID of the client, which is assigned by the server
        when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: The configuration
        data: The data owned by the client
        model: The model maintained locally
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
        ss_manager: secret sharing manager
        msg_buffer: dict buffer for storing message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        # 调用基类 BaseClient 的初始化方法，保存 ID、状态、配置、模型和策略
        super(Client, self).__init__(ID, state, config, model, strategy)

        self.data = data

        # Register message handlers
        # 注册默认的消息处理函数 (Handlers)
        # 将消息类型（如 'model_para'）映射到对应的回调函数（如 callback_funcs_for_model_para）
        self._register_default_handlers()

        # Un-configured worker # 如果没有配置，直接返回（通常用于未配置的 Worker 占位）
        if config is None:
            return

        # the unseen_client indicates that whether this client contributes to
        # FL process by training on its local data and uploading the local
        # model update, which is useful for check the participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        # 设置 is_unseen_client 标志
        # 该标志用于指示此客户端是否作为"未见"客户端存在。
        # 未见客户端不参与训练过程（不上传模型更新），主要用于检查联邦学习中的参与泛化差距
        self.is_unseen_client = is_unseen_client

        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        # 这部分逻辑用于模拟联邦学习中的对抗攻击（Adversarial Attack）
        # 解析攻击者 ID，支持单个 int 或 list 格式
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attack_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        # Attack only support the stand alone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        # 判断当前客户端是否为攻击者
        # 条件：ID 在攻击者列表中，且配置了攻击方法，且处于单机模拟模式 (standalone)
        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        # 使用工厂函数 get_trainer 构建训练器实例
        # 训练器负责具体的本地训练循环（如前向传播、反向传播）
        # 这里会将 is_attacker 传入，以便 Trainer 加载攻击相关的逻辑
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)
        self.device = device

        # For client-side evaluation
        # 初始化用于存储评估结果的字典
        self.best_results = dict()
        self.history_results = dict()
        # in local or global training mode, we do use the early stopper.
        # Otherwise, we set patience=0 to deactivate the local early-stopper
        # 配置早停 (Early Stopper) 的耐心值 (patience)
        # 如果是 "local" (本地训练) 或 "global" (集中式训练) 模式，使用配置的 patience；
        # 否则（标准联邦学习），将 patience 设为 0，意味着客户端不进行本地早停，而是听从服务器指令
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in [
                "local", "global"
            ] else 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)

        # Secret Sharing Manager and message buffer
        # 初始化加法秘密共享管理器 (AdditiveSecretSharing)
        # 仅在配置开启 use_ss 时初始化，用于安全聚合
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None
        # 初始化消息缓冲区，用于存储分片消息
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Communication and communication ability
        # 用于在单机模拟模式下估算通信和计算的时间成本
        # 从 kwargs 中提取资源信息（计算速度和带宽）
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None:
            self.comp_speed = float(
                kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(
                kwargs['resource_info']['communication'])  # (kbit/s)
        else:
            self.comp_speed = None
            self.comm_bandwidth = None

        # 计算模型大小 (Model Size) 用于预估通信延迟
        if self._cfg.backend == 'torch':
            try:
                # 使用 pickle 序列化模型来计算大小 (kbits)
                self.model_size = sys.getsizeof(pickle.dumps(
                    self.model)) / 1024.0 * 8.  # kbits
            except Exception as error:
                self.model_size = 1.0
                logger.warning(f'{error} in calculate model size.')
        else:
            # TODO: calculate model size for TF Model
            # TensorFlow 后端暂未实现模型大小计算
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:'
                           f'{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id
        if self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue'] # 获取共享队列
            # 根据进程数选择通信管理器 
            if self._cfg.federate.process_num <= 1:
                # 单进程模拟：使用基于内存队列的管理器
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            else:
                # 多进程模拟 (DDP)：使用支持分布式数据并行的管理器
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            self.local_address = None
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            server_host = kwargs['server_host']
            server_port = kwargs['server_port']
            self.comm_manager = gRPCCommManager(
                host=host,
                port=port,
                client_num=self._cfg.federate.client_num,
                cfg=self._cfg.distribute)
            logger.info('Client: Listen to {}:{}...'.format(host, port))
            self.comm_manager.add_neighbors(neighbor_id=server_id,
                                            address={
                                                'host': server_host,
                                                'port': server_port
                                            })
            self.local_address = {
                'host': self.comm_manager.host,
                'port': self.comm_manager.port
            }

    # 在单机模拟 (Standalone Mode) 环境下，计算和生成模拟的时间戳
    def _gen_timestamp(self, init_timestamp, instance_number):
        # 如果初始时间戳为空（通常发生在初始化阶段或非时间敏感的任务），直接返回 None
        if init_timestamp is None:
            return None
        # 计算计算延迟 (comp_cost) 和 通信延迟 (comm_cost)
        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number, # 样本数量：训练的数据量越大，计算越慢
            comm_size=self.model_size, # 模型大小：模型越大，传输越慢
            comp_speed=self.comp_speed, # 计算速度：模拟设备的硬件性能 (秒/样本)，计算开销为样本数*单样本处理时间
            comm_bandwidth=self.comm_bandwidth) # 通信带宽：模拟网络环境 (kbit/s)，通信开销为模型大小 / 带宽
        # 最终时间 = 初始时间 (Server发来的时间) + 计算耗时 + 传输耗时
        # 这模拟了Server收到该客户端更新时的逻辑时间点
        return init_timestamp + comp_cost + comm_cost

    # 计算模型更新量（Delta），即用训练后的模型参数减去训练前的初始模型参数
    def _calculate_model_delta(self, init_model, updated_model):
        # 这个功能在异步联邦学习或需要鲁棒聚合算法（如 Krum, Median）的场景中非常关键
        # 因为服务器需要的是“梯度的近似值”或“更新方向”，而不是最终的参数值
        # 检查 init_model 是否为列表
        # 如果不是（即单模型情况，通常是一个 state_dict 字典），将其包装成一个单元素列表
        if not isinstance(init_model, list): 
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            # 使用深拷贝 (deep copy) 创建一个与初始模型结构完全相同的对象作为 Delta 容器
            # 这样做可以保留原有 Tensor 的属性（如设备、数据类型），同时避免修改原始引用
            model_delta = copy.deepcopy(init_model[model_index])
            # 遍历模型的所有参数键 (如 'conv1.weight', 'fc.bias')
            for key in init_model[model_index].keys():
                # 计算更新量：训练后的参数值 - 初始参数值
                model_delta[key] = updated_model[model_index][
                    key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        # 如果原输入是单模型，返回单个字典 (解包)
        else:
            return model_deltas[0]

    # 向服务器发送一个“注册申请”，告诉服务器“我上线了，请带我一起训练”
    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        # 在单机模式下，这会将消息放入共享内存队列中
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,  # 时间戳：握手阶段通常设为 0
                    content=self.local_address)) # 内容：客户端的通信地址

    # 持续监听来自服务器的消息并做出响应
    def run(self):
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        while True: # 开启一个死循环。这确保客户端进程一直保持活跃状态，不断地等待服务器的指令。只有收到特定的结束信号（如 'finish'），循环才会终止
            # 在分布式模式下，这是一个阻塞调用，客户端会挂起，直到通过网络收到服务器发来的数据包
            # 在单机模拟模式下，它是从共享的消息队列中取出一条消息
            msg = self.comm_manager.receive()
            if self.state <= msg.state: # 当收到的消息轮次大于或等于客户端当前轮次时，才会被处理
                self.msg_handlers[msg.msg_type](msg) # 根据消息类型找到对应的回调函数

            if msg.msg_type == 'finish':
                break
    
    # 在特定环境（如单进程或多进程的单机模拟）下启动客户端
    def run_standalone(self):
        """
        Run in standalone mode
        """
        self.join_in()
        self.run()

    # 负责接收全局模型、执行本地训练、并上传更新结果
    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        # 分支一：处理秘密共享片段
        if 'ss' in message.msg_type: # 这是来自其他客户端的秘密分片，而不是来自服务器的完整模型
            # A fragment of the shared secret
            # 提取轮数、内容和时间戳
            state, content, timestamp = message.state, message.content, \
                                        message.timestamp
            # 将收到的分片存入训练缓冲区
            self.msg_buffer['train'][state].append(content)

            # 检查是否收集齐了所有其他客户端发来的分片
            # client_num 在这里通常指参与秘密共享的一组客户端数量
            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                # --- 聚合分片逻辑 ---
                model_list = self.msg_buffer['train'][state]
                # 获取样本大小和第一个分片作为基准
                sample_size, first_aggregate_model_para = model_list[0]
                # 处理单模型与多模型的情况
                single_model_case = True
                if isinstance(first_aggregate_model_para, list): # 多模型
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else: # 单模型
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should " \
                        "a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                # 对所有分片进行累加 (Additive Secret Sharing 的恢复过程)
                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                # 将聚合后的部分和（Partial Sum）发送给服务器
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        # 分支二：标准联邦学习流程
        else:
            round = message.state
            sender = message.sender
            timestamp = message.timestamp
            content = message.content # 服务器发来的模型参数

            # dequantization
            # 1. 解量化 (Dequantization)
            # 如果配置了 uniform 量化，需要先解压参数
            if self._cfg.quantization.method == 'uniform':
                from federatedscope.core.compression import \
                    symmetric_uniform_dequantization
                if isinstance(content, list):  # multiple model
                    content = [
                        symmetric_uniform_dequantization(x) for x in content
                    ]
                else:
                    content = symmetric_uniform_dequantization(content)

            # When clients share the local model, we must set strict=True to
            # ensure all the model params (which might be updated by other
            # clients in the previous local training process) are overwritten
            # and synchronized with the received model
            # 2. 设备同步 (DDP 多进程模式)
            # 确保参数在正确的设备上 (GPU/CPU)
            if self._cfg.federate.process_num > 1:
                for k, v in content.items():
                    content[k] = v.to(self.device)
            # 3. 加载全局模型到本地 Trainer
            # strict=True/False 决定是否允许部分参数缺失
            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model)
            self.state = round
            # B 检查是否因早停而跳过训练
            skip_train_isolated_or_global_mode = \
                self.early_stopper.early_stopped and \
                self._cfg.federate.method in ["local", "global"]
            # 如果是"未见客户端" (只做测试不训练) 或 已经早停
            if self.is_unseen_client or skip_train_isolated_or_global_mode:
                # for these cases (1) unseen client (2) isolated_global_mode,
                # we do not local train and upload local model
                # 设置空的返回结果，不进行train()
                sample_size, model_para_all, results = \
                    0, self.trainer.get_model_para(), {}
                if skip_train_isolated_or_global_mode:
                    logger.info(
                        f"[Local/Global mode] Client #{self.ID} has been "
                        f"early stopped, we will skip the local training")
                    self._monitor.local_converged()
            # C. 执行本地训练
            else: # 如果在普通 FL 模式下早停，发出警告 (可能产生负面影响)
                if self.early_stopper.early_stopped and \
                        self._monitor.local_convergence_round == 0:
                    logger.info(
                        f"[Normal FL Mode] Client #{self.ID} has been locally "
                        f"early stopped. "
                        f"The next FL update may result in negative effect")
                    self._monitor.local_converged()
                # 【核心步骤】调用 Trainer 执行本地训练
                # 返回：样本数, 训练后的参数, 评估结果
                sample_size, model_para_all, results = self.trainer.train()
                # 如果共享本地模型且非在线聚合，需要深拷贝一份参数，防止引用被修改
                if self._cfg.federate.share_local_model and not \
                        self._cfg.federate.online_aggr:
                    model_para_all = copy.deepcopy(model_para_all)
                # 记录并保存训练日志 (WandB 支持)
                train_log_res = self._monitor.format_eval_res(
                    results,
                    rnd=self.state,
                    role='Client #{}'.format(self.ID),
                    return_raw=True)
                logger.info(train_log_res)
                if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                    self._monitor.save_formatted_results(train_log_res,
                                                         save_file_name="")

            # Return the feedbacks to the server after local update
            # D. 返回结果给服务器
            if self._cfg.federate.use_ss:
                # ... (一系列类型断言检查) ...
                assert not self.is_unseen_client, \
                    "Un-support using secret sharing for unseen clients." \
                    "i.e., you set cfg.federate.use_ss=True and " \
                    "cfg.federate.unseen_clients_rate in (0, 1)"
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of " \
                        "multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    # 加权：参数 * 样本数
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    # 切分：将参数切分为多个秘密分片
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                # 分发：遍历所有邻居 (其他 Client)，发送对应的分片 'ss_model_para'
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        # ... (提取对应 neighbor 的分片) ...
                        content_frame = model_para_list_all[0][frame_idx] if \
                            single_model_case else \
                            [model_para_list[frame_idx] for model_para_list
                             in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    timestamp=self._gen_timestamp(
                                        init_timestamp=timestamp,
                                        instance_number=sample_size),
                                    content=content_frame))
                        frame_idx += 1
                # 同时也需要把自己的一份分片存入本地 msg_buffer，用于后续聚合
                content_frame = model_para_list_all[0][frame_idx] if \
                    single_model_case else \
                    [model_para_list[frame_idx] for model_para_list in
                     model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else:
                # 如果是异步训练 或 使用了鲁棒聚合规则 (如 Krum, Median)
                if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                        ['krum', 'normbounding', 'median', 'trimmedmean',
                         'bulyan']:
                    # Return the model delta when using asynchronous training
                    # protocol, because the staled updated might be discounted
                    # and cause that the sum of the aggregated weights might
                    # not be equal to 1
                    # 计算模型增量 (Delta) = 新模型 - 旧模型
                    # 而不是直接发送新模型
                    shared_model_para = self._calculate_model_delta(
                        init_model=content, updated_model=model_para_all)
                else:
                    # 默认情况：发送完整的训练后模型
                    shared_model_para = model_para_all

                # quantization # 量化 (Quantization)
                if self._cfg.quantization.method == 'uniform':
                    from federatedscope.core.compression import \
                        symmetric_uniform_quantization
                    nbits = self._cfg.quantization.nbits
                    if isinstance(shared_model_para, list):
                        shared_model_para = [
                            symmetric_uniform_quantization(x, nbits)
                            for x in shared_model_para
                        ]
                    else:
                        shared_model_para = symmetric_uniform_quantization(
                            shared_model_para, nbits)
                # 发送最终结果给服务器
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self._gen_timestamp( # 生成模拟时间戳 (包含计算和通信延迟)
                                init_timestamp=timestamp,
                                instance_number=sample_size),
                            content=(sample_size, shared_model_para)))

    # 处理服务器分配给客户端的ID
    def callback_funcs_for_assign_id(self, message: Message):
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.
        # 在分布式联邦学习的握手阶段，客户端启动时可能尚未拥有唯一的 ID（通常初始化为 -1）。
        当它向服务器发送加入请求后，服务器会为其分配一个唯一的 ID 并发回。这个函数就是用来接收并应用这个新 ID 的
        Arguments:
            message: The received message
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    # 响应服务器的信息查询请求
    def callback_funcs_for_join_in_info(self, message: Message):
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        # 提取服务器请求的内容，这是一个字符串列表，例如 ['num_sample', 'client_resource']
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict() # 初始化一个字典，用于存放即将回传给服务器的信息
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    # --- 情况 1：基于 Batch 的更新 ---
                    # 如果配置是按 Batch 更新，样本量 = 本地更新步数 * Batch Size
                    num_sample = self._cfg.train.local_update_steps * \
                                 self._cfg.dataloader.batch_size
                else:
                    # --- 情况 2：基于 Epoch 的更新 (默认情况) ---
                    # 如果配置是按 Epoch 更新，样本量 = 本地更新轮数 * 数据集总长度
                    num_sample = self._cfg.train.local_update_steps * \
                                 len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                # --- 特殊情况：全图训练 ---
                # 如果是 NodeFullBatchTrainer (通常用于图神经网络的全图训练)
                # 样本数量修正为图中的节点数量 (x.shape[0])
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = \
                        self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                # 确保已初始化带宽和计算速度参数 (通常在单机模拟模式下通过 kwargs 传入)
                assert self.comm_bandwidth is not None and self.comp_speed \
                       is not None, "The requirement join_in_info " \
                                    "'client_resource' does not exist."
                # 计算预估的总延迟成本
                # 公式：通信延迟 (模型大小 / 带宽) + 计算延迟 (comp_speed)
                join_in_info['client_resource'] = self.model_size / \
                    self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    # 接收并注册其他客户端的通信地址，从而在本地构建出网络拓扑
    def callback_funcs_for_address(self, message: Message):
        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology
        这通常用于需要客户端之间直接通信（P2P）的高级场景，例如安全聚合（Secret Sharing）或纵向联邦学习（Vertical FL），
        在这些场景中，客户端不仅连接服务器，还需要连接其他客户端。
        Arguments:
            message: The received message
        """
        # 提取消息内容。
        # content 是一个字典，结构通常为 {client_id: {host: ip, port: port}, ...}
        # 它包含了当前联邦网络中所有参与者的地址信息。
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                # --- 注册邻居 ---
                # 调用通信管理器，将这个 ID 和对应的地址添加到本地的邻居列表 (neighbors) 中。
                # 这一步完成后，self.comm_manager.send(..., receiver=[neighbor_id]) 
                # 就能查找到正确的 IP 和端口发送数据了。
                self.comm_manager.add_neighbors(neighbor_id, address)

    # 响应服务器发来的评估 (Evaluate) 请求
    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        # 提取发送者 (Server) 和时间戳
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        # 如果消息中包含模型参数 (content is not None)
        if message.content is not None:
            logger.info(111)
            # 将参数加载到本地 Trainer 中
            # strict参数控制加载的严格程度：如果启用了本地模型共享(share_local_model)，通常需要更严格的匹配
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        # 条件：已触发早停且模式是 'local' 或 'global'，如果这些模式下已经触发早停，就不再重复计算评估指标
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global"
        ]:  
            # 直接复用之前记录的最佳结果，不再运行 evaluate
            metrics = list(self.best_results.values())[0]
            logger.info(222)
        else:
            logger.info(333)
            metrics = {}
            # --- 步骤 A: 评估前微调 (Test-time Finetuning) ---
            # 某些个性化联邦学习算法 (如 pFedMe) 要求在评估前对模型进行少量微调
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            # --- 步骤 B: 遍历数据集切分 ---
            # 遍历配置中指定的数据集 (通常是 ['test'], 也可以包含 ['val'])
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                # 调用 Trainer 执行真正的评估计算 (前向传播、计算 Loss/Acc)
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                # 分布式模式下打印日志
                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))
                # 将结果合并到总指标字典中
                metrics.update(**eval_metrics)

            # --- 步骤 C: 结果格式化与记录 ---
            # 格式化结果以便日志输出
            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            logger.info(formatted_eval_res)
            # --- 步骤 D: 更新最佳结果 ---
            # update_best_this_round: 布尔值，表示本轮是否创造了新的最佳记录
            update_best_this_round = self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res['Results_raw'],
                results_type=f"client #{self.ID}",
            )

            # --- 步骤 E: 保存最佳模型 ---
            # 如果本轮效果最好 且 配置开启了保存客户端模型
            if update_best_this_round and self._cfg.federate.save_client_model:
                path = add_prefix_to_path(f'client_{self.ID}_',
                                          self._cfg.federate.save_to)
                # 仅主进程执行保存操作
                if self.ds_rank == 0:
                    self.trainer.save_model(path, self.state)

            # --- 步骤 F: 历史记录与早停检查 ---
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            # 检查是否满足早停条件
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    # 客户端生命周期的终结者。当服务器决定结束训练（无论是正常跑完所有轮次，还是触发早停）时，会发送finish消息
    def callback_funcs_for_finish(self, message: Message):
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        # 打印醒目的日志，表明客户端 ID 已收到结束信号
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        # 服务器可能会在 finish 消息中附带最终的全局模型（或者历史最佳模型）。
        # 客户端需要将其加载到本地，以确保本地模型与服务器保持一致（例如用于后续的本地测试或保存）
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        # 调用 Monitor 组件的结束方法
        # 这通常涉及：
        # 1. 关闭日志文件句柄
        # 2. 结束 WandB / TensorBoard 的运行 (Run)
        # 3. 打印最终的统计摘要
        self._monitor.finish_fl()

    # 当服务器端的早停机制（Early Stopper）检测到模型性能不再提升时，会向所有客户端广播收敛信号
    def callback_funcs_for_converged(self, message: Message):
        """
        The handling function for receiving the signal that the FL course \
        converged
        处理 converged 类型的消息。这意味着整个联邦学习任务因为满足了早停条件
        （例如 Loss 连续 N 轮没有下降）而提前结束，而不是因为跑完了所有轮次

        Arguments:
            message: The received message
        """
        self._monitor.global_converged()

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
