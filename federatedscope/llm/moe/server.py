import logging
import torch
import random
import math
from federatedscope.core.message import Message
import os
import copy

from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_param_dict
from federatedscope.llm.moe.router import PeMoERouter
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict, add_prefix_to_path, get_ds_rank

logger = logging.getLogger(__name__)


class LLMMultiLoRAServer(Server):
    """
    Server implementation
    We broadcast the model to each client and ask them to train locally
    Afterward, we collect the model back and save it as checkpoints
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
                 **kwargs):
        super(LLMMultiLoRAServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
    
        # === 1. 初始化并加载 Router ===
        input_dim = self._cfg.llm.hidden_size
        num_experts = self._cfg.llm.adapter.count
        self.device = device
        self.router = PeMoERouter(input_dim, num_experts).to(self.device)

        logger.info(f"Server loading initial router from {self._cfg.federate.save_to_router}")
        self.router.load_state_dict(torch.load(self._cfg.federate.save_to_router, map_location=self.device))


    def _register_default_handlers(self): # 向FederatedScope的消息处理系统注册自定义的消息处理函数
        # 1. 保留父类的默认处理逻辑（处理 join, train, eval 等标准消息）
        super()._register_default_handlers()

    # 每一轮训练结束时，负责收集所有客户端上传的模型更新，并聚合成新的全局模型参数
    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num): # 通常情况下model_num = 1
            model = self.models[model_idx] # 获取模型
            aggregator = self.aggregators[model_idx] # 获取聚合器
            msg_list = list()

            for client_id in train_msg_buffer.keys(): # 遍历每个client发过来的
                if self.model_num == 1: # 单模型，直接将发过来的全部content添加
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))
            # 处理过时消息
            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }
            # logger.info(f'The staleness is {staleness}')
            if self._cfg.llm.adapter.moe:
                avg_model, avg_router = aggregator.aggregate_on_moe(agg_info)
            # 分支 2: 正式分组训练阶段 (Phase 2 - FedBiscuit)
            else:
                raise NotImplementedError("aggregation may be error.")

            # Due to lazy load, we merge two state dict
            # 将聚合结果合并回全局模型
            merged_param = merge_param_dict(model.state_dict().copy(), avg_model)
            model.load_state_dict(merged_param, strict=False)
            
            logger.info("The model has been aggregation!")

            warmup_round = self._cfg.llm.adapter.warmup.round
            # 总预热轮数
            total_warmup_round = \
                        warmup_round * self._cfg.llm.adapter.count
            # total_warmup_round = 110
            # 获取路由网络更新间隔r
            r = self._cfg.llm.adapter.grouping.round
            if self.state >= total_warmup_round and \
                    (self.state - total_warmup_round) % r == 0:
                merged_router_param = merge_param_dict(self.router.state_dict().copy(), avg_router)
                self.router.load_state_dict(merged_router_param, strict=False)
                path = add_prefix_to_path(f'{self.state}_',
                                        self._cfg.federate.save_to_final_router)
                logger.info("The router has been saved !")
                torch.save(self.router.state_dict(), path)

            # 保存最终模型
            if self.state % self._cfg.federate.save_freq == 0 and \
                    self._cfg.federate.save_freq > 0:
                # 构造保存路径，例如 "100_model.pth"
                path = add_prefix_to_path(f'{self.state}_',
                                        self._cfg.federate.save_to)
                # 只有主进程 (rank 0) 执行保存操作，避免多进程写入冲突

                # if self.ds_rank == 0:
                #     self.aggregator.save_model(path, self.state)
                logger.info("The model has been saved !")
                self.aggregator.save_model(path, self.state)

        return aggregated_num
    
    # 启动入口
    def trigger_for_start(self):
        # start feature engineering (This part is for hard code)
        # 1. 如果所有预期的客户端都加入
        if self.check_client_join_in():
            logger.info('Waited all clients join, start now...')
            # 2. 触发初始广播 (关键修改点)
            self.trigger_for_feat_engr(self.broadcast_model_para, {
                'msg_type': 'model_para',  # 强制发送评估指令，而非训练指令
                'sample_client_num': self.sample_client_num,
                'filter_unseen_clients': False,  # 确保发给所有已连接的客户端
            })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))

    # 【新修改】Server 向 Client 分发全局模型，加入router参数
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
        all_model_para = dict()
        router_para = torch.load(self._cfg.federate.save_to_router, map_location=self.device)
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
        all_model_para['model_para'] = model_para
        all_model_para['router_para'] = router_para

        """
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
        """
        # We define the evaluation happens at the end of an epoch
        # 6. 发送消息
        # 计算消息中的轮次 (state)
        # 如果是评估消息 (evaluate)，通常指的是评估"上一轮"训练后的模型，所以 state - 1
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state
        logger.info(f"The receiver is {receiver}")
        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver, # 采样的全部客户端
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=all_model_para))
        # 如果是在线聚合 (Online Aggregation)，广播后需要重置聚合器状态
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        # 恢复采样器状态：将之前标记为 'unseen' 的客户端恢复为 'seen'
        # 以免影响后续逻辑
        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

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
        if self._cfg.federate.save_to_router != '':
            torch.save(self.router.state_dict(), self._cfg.federate.save_to_final_router)

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

