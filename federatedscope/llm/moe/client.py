import copy
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.data import ClientData
from federatedscope.llm.moe.router import PeMoERouter
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.dataloader import LLMDataCollator
from federatedscope.llm.dataset.llm_dataset import DefaultToken

logger = logging.getLogger(__name__)


def _get_or_generate_embeddings(data: ClientData, device):
    """
    获取本地数据的 Embeddings。
    如果数据集里已经预处理了 embedding 则直接读取，否则通过 base model 生成。
    """

    if hasattr(data.train_data, '__getitem__') and \
        isinstance(data.train_data[0], dict) and \
        'embedding' in data.train_data[0]:
        
        # 直接提取
        embeddings_train = [item['embedding'] for item in data.train_data]
        embeddings_val = [item['embedding'] for item in data.val_data]
        embeddings_test = [item['embedding'] for item in data.test_data]
        embeddings1 = torch.tensor(np.array(embeddings_train), dtype=torch.float32).to(device)
        embeddings2 = torch.tensor(np.array(embeddings_val), dtype=torch.float32).to(device)
        embeddings3 = torch.tensor(np.array(embeddings_test), dtype=torch.float32).to(device)
        # new_train_data = [{k: v for k, v in d.items() if k != 'embedding'} for d in data.train_data]
        # new_val_data = [{k: v for k, v in d.items() if k != 'embedding'} for d in data.val_data]
        # new_test_data = [{k: v for k, v in d.items() if k != 'embedding'} for d in data.test_data]
        # return [embeddings1, embeddings2, embeddings3], ClientData(data.client_cfg, new_train_data, new_val_data, new_test_data)
        return [embeddings1, embeddings2, embeddings3]
    else:
        raise NotImplementedError("Embeddings not found in dataset. Please ensure data preprocessing includes embedding generation.")

class LLMMultiLoRAClient(Client):
    """
    Client implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
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
                 *args,
                 **kwargs):
        # 
        self.embedding_all = _get_or_generate_embeddings(data, device)

        super(LLMMultiLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

        # === 1. 初始化Router ===
        input_dim = self._cfg.llm.hidden_size
        num_experts = self._cfg.llm.adapter.count
        self.device = device
        self.router = PeMoERouter(input_dim, num_experts).to(self.device)

        model_name, _ = self._cfg.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, self._cfg.data.root,
                                          self._cfg.llm.tok_len)
        

        try:
            self.choices = []
            for choice in self._cfg.trainer.choices:
                self.choices.append(
                    self.tokenizer(f': {choice}')['input_ids'][-1])
                # assert len(self.choices[-1]) == 1
            logger.info(f'Choice indices: {self.choices}')
        except AssertionError:
            raise AssertionError('The choice should be limited to one token.')
        except:
            raise ValueError('trainer.choices not found in the config.')

    # 注册自定义的消息处理器
    def _register_default_handlers(self):
        super()._register_default_handlers()

    # 客户端接收到服务端发来模型参数，开始训练（激活并训练某一个Selector）
    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                # 如果是在多进程环境下，确保数据移动到正确的设备
                content[k] = v.to(self.device)
        
        model_para = content['model_para']
        router_para = content['router_para']

        self.router.load_state_dict(router_para)
        self.state = round
        
        # 如果在普通 FL 模式下早停，发出警告 (可能产生负面影响)
        if self.early_stopper.early_stopped and \
                self._monitor.local_convergence_round == 0:
            logger.info(
                f"[Normal FL Mode] Client #{self.ID} has been locally "
                f"early stopped. "
                f"The next FL update may result in negative effect")
            self._monitor.local_converged()

        clientDataAll = self._get_data_from_embeddings()

        model_para_all_moe = [] # 多个选择器的参数
        warmup_round = self._cfg.llm.adapter.warmup.round
        # 总预热轮数
        total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count
        # total_warmup_round = 110
        # 获取路由网络更新间隔r
        r = self._cfg.llm.adapter.grouping.round
        for i in range(self._cfg.llm.adapter.count):
            clientData = clientDataAll[i]
            # 查看embedding是否去除干净
            if len(clientData.train_data) == 0:
                continue
            self.trainer_copy = get_trainer(model=self.model,
                                            data=clientData,
                                            device=self.device,
                                            config=self._cfg,
                                            is_attacker=self.is_attacker,
                                            monitor=self._monitor)
            self.trainer_copy.update(model_para,
                            strict=self._cfg.federate.share_local_model)
            self.model.set_active_adapter(f'Adapter_{i}')
            self.model.train() 
            # 通过self.data中的embedding和路由网络获取每个选择器对应的self.newdata
            #执行标准的本地训练循环
            sample_size, model_para_all, results = self.trainer_copy.train()
            # 保存每一个选择器训练后的权重
            model_para = copy.deepcopy(model_para_all)
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
                # 自身选择器模型参数
                model_para_all = {
                    key: value
                    for key, value in model_para_all.items()
                    if f'Adapter_{i}.' in key
                }
            model_para_all_moe.append((sample_size, model_para_all))
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                        save_file_name="")
        # 是否使用warmup
        if self.state >= total_warmup_round and \
                    (self.state - total_warmup_round) % r == 0:
                router_para = self._train_router_locally()
        
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                  instance_number=sample_size),
                    content=(model_para_all_moe, (len(self.data.train_data), router_para))))

    

    # 【新修改】从embedding中预测，从而重组每个选择器的数据
    def _get_data_from_embeddings(self):
        preds_all = [[],[],[]]
        with torch.no_grad():
            for i in range(3):
                embeddings = self.embedding_all[i]
                if embeddings.numel() == 0:
                    continue
                logits = self.router(embeddings)
                preds = torch.argmax(logits, dim=1)
                preds_all[i] = preds
        
        local_partitions1 = {exp_id: [] for exp_id in range(self._cfg.llm.adapter.count)}
        local_partitions2 = {exp_id: [] for exp_id in range(self._cfg.llm.adapter.count)}
        local_partitions3 = {exp_id: [] for exp_id in range(self._cfg.llm.adapter.count)}
        for i in range(len(preds_all[0])):
            local_partitions1[preds_all[0][i].item()].append(self.data.train_data[i])
        for i in range(len(preds_all[1])):
            local_partitions2[preds_all[1][i].item()].append(self.data.val_data[i])
        for i in range(len(preds_all[2])):
            local_partitions3[preds_all[2][i].item()].append(self.data.test_data[i])
        clientData = []
        for i in range(self._cfg.llm.adapter.count):
            clientData.append(ClientData(self.data.client_cfg, local_partitions1[i], local_partitions2[i], local_partitions3[i]))
        return clientData
    
    # 训练路由网络，得到更新的参数
    def _train_router_locally(self):
        """
        实现基于性能蒸馏的路由损失函数训练
        """
        train_data = self.data.train_data
        self.router.train()
        
        # 此时选择器已训练完，计算每个样本在所有专家下的 Loss
        
        # 1. 构造监督标签：寻找每个样本真正的最佳专家
        best_expert_labels = []
        
        dataloader = DataLoader(
            dataset=train_data,
            batch_size=10,
            shuffle=False,
            collate_fn=LLMDataCollator(tokenizer=self.tokenizer),
        )

        for idx, data_batch in enumerate(tqdm(dataloader)):
            input_ids = data_batch["input_ids"].to("cuda:0")
            labels = data_batch["labels"].to("cuda:0")
            attention_mask = data_batch["attention_mask"].to("cuda:0")
            collective_choices = []
            for exp_id in range(self._cfg.llm.adapter.count):
                with torch.no_grad():
                    self.model.set_active_adapter(f"Adapter_{exp_id}")
                    self.model.eval()
                    outputs = self.model(input_ids=input_ids,
                                        labels=labels,
                                        attention_mask=attention_mask)
                _, _, loss = self.cal_loss(logits=outputs.logits, labels=labels, choices=self.choices)
                collective_choices.append(loss.tolist())
            array = np.array(collective_choices).T
            best_expert_labels += [
                    np.argmin(array[i])
                    for i in range(len(array))
                ]
        
        X = self.embedding_all[0]
        y = torch.tensor(np.array(best_expert_labels), dtype=torch.long).to(self.device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.router.parameters(), lr=1e-3, weight_decay=1e-4)
        self.router.train()
        epochs = 1000
        # 2. 更新 Router 参数
        # 损失函数 = 性能预测交叉熵 + 负载均衡损失 (可选)
        for epoch in range(epochs): # 局部微调几轮
            total_loss = 0
            correct = 0
            total = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                # Forward
                logits = self.router(batch_X)
                # Loss
                loss = criterion(logits, batch_y)
                # Backward
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # 计算准确率 (Accuracy) 用于监控
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            avg_loss = total_loss / len(loader)
            acc = correct / total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Router Pre-train Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2%}")
            if acc >= 0.95:
                break
            
        return {k: v.cpu() for k, v in self.router.state_dict().items()}

    def cal_loss(self, logits, labels, choices):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
        for idx, choice in enumerate(choices):
            new_labels[shift_labels == choice] = idx

        # new_logits = logits * mask.unsqueeze(-1)
        # new_logits = torch.sum(new_logits, dim=1)[:, choices]
        # new_labels = torch.sum(new_labels, dim=1)

        new_logits = shift_logits[..., choices]
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=DefaultToken.IGNORE_INDEX.value)
        # 维度(Batch * (SeqLen-1))
        token_losses = loss_fn(new_logits.view(-1, len(choices)), new_labels.view(-1))
        batch_size = labels.size(0)
        token_losses = token_losses.view(batch_size, -1)
        sample_losses = token_losses.sum(dim=1)

        # return new_logits.view(-1, len(choices)), new_labels.view(-1), loss
        return new_logits, new_labels, sample_losses

