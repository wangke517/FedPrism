import os
import sys
import argparse
import gc
import json
import logging
import random
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.data import ClientData
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.dataloader.dataloader import get_tokenizer
# 导入 standalone_training 中的类和函数以便复用
from federatedscope.llm.dataset.llm_dataset import (
    DefaultToken,
    LLMDataset,
    LLMComparisonDataset,
)
from federatedscope.llm.rlhf.standalone_training import \
    RLHF_finetuning, get_rlhf_prompts_dataset, get_input_data
from federatedscope.llm.trainer.reward_trainer import DPORewardTrainer

from federatedscope.llm.moe.router import PeMoERouter

logger = logging.getLogger(__name__)

class PeMoEInitializer(RLHF_finetuning):
    def __init__(self, model, tokenizer, config, selector_model, selector_tokenizer, generator_tokenizer, device="cpu", **kwargs):
        # 初始化父类，传入所有必要的模型和tokenizer
        super().__init__(
            model=model, 
            tokenizer=tokenizer, 
            config=config, 
            selector_model=selector_model, 
            selector_tokenizer=selector_tokenizer, 
            generator_tokenizer=generator_tokenizer,
            device=device,
            **kwargs
        )
        self.selector_cfg = kwargs['selector_cfg']
        self.num_experts = None if self.selector_cfg is None else self.selector_cfg.llm.adapter.count
        self.selector_model = selector_model
        self.selector_tokenizer = selector_tokenizer
        
        
    def run_initialization(self):
        logger.info("Starting Initialization Phase...")

        # 1. 生成 Pairwise 数据 (使用 Generator Model)
        # 复用父类方法 [cite: 3]
        logger.info("Generating pairwise responses using Generator...")

        # 下载指令，通过LLM得到对应的两个回复对
        self.raw_pairwise_data = self.load_pairwise_data()

        # 获得对应的embedding
        list_data_dict = self.load_embeddings()
        #  随机选择最佳答案 (伪标签)
        logger.info("Applying Random Selection to create pseudo-labels...")
        labeled_data = self._apply_random_preference(list_data_dict)
        
        
        # 释放 Generator 显存
        logger.info("Offloading Generator Model to free VRAM...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'generator_tokenizer'):
            del self.generator_tokenizer
        
        # 强制垃圾回收和清空 CUDA 缓存
        gc.collect()
        torch.cuda.empty_cache()

        # 基于语义聚类划分数据
        logger.info("Partitioning data for Experts using K-Means...")
        partitioned_data, embeddings_shape = self._partition_data_by_clustering(labeled_data)
        
        # 初始化训练每个 Expert (使用 Selector Model)
        logger.info("Pre-training Experts (Selector Adapters) on partitioned data...")
        self._pretrain_experts(partitioned_data, self.selector_cfg, self.selector_tokenizer, self.selector_prompt)

        # 初始化训练 Router
        logger.info("Pre-training Router...")
        router_save_path = self._pretrain_router(partitioned_data, embeddings_shape)

        logger.info("Initialization Complete.")
        return self.selector_model, self.router, embeddings_shape, router_save_path

    def _apply_random_preference(self, list_data_dict):
        """随机分配偏好，模拟初始化差异"""
        for sample in list_data_dict:
            sample["choice"] = random.choice([0, 1])
            sample['choice'] = " " + chr(sample['choice'] + ord("A"))
            sample.pop("fake_choice", None)
        return list_data_dict

    def load_embeddings(self, save_path=None):
        """
        类似于 load_pairwise_data，检查是否存在缓存的 embedding json 文件。
        - 如果存在：加载并自动将 embedding 字段转回 numpy 数组。
        - 如果不存在：调用 _get_batch_embeddings 生成，并保存为 json。
        """
        # 1. 如果没有指定路径，自动构造默认路径
        if save_path is None:
            # 提取模型名和数据集名用于文件名
            _, model_name = self.config.model.type.split("@")[0].split('/', 1)
            dataset_name, _ = self.config.data.type.split("@")
            num_comp = max(2, self.config.llm.num_completions)
            # 构造文件名，例如: rlhf_embeddings_Llama-2-7b_shp-rlhf.json
            save_path = os.path.join(
                self.data_root,
                f"rlhf_embeddings_{model_name}_{dataset_name}_{num_comp}.json"
            )

        # 2. 检查文件是否存在
        if os.path.exists(save_path):
            logger.info(f"Loading cached embeddings from {save_path}")
            try:
                # 加载 JSON
                list_data_dict = json.load(open(save_path, "r"))
                logger.info(f"Successfully loaded {len(list_data_dict)} embeddings.")
                return list_data_dict
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Re-generating...")

        # 3. 如果文件不存在（或加载失败），则重新生成
        logger.info("Cached embeddings not found. Generating new embeddings...")
        
        # 调用之前写好的生成函数
        list_data_dict = self._get_batch_embeddings(
            self.raw_pairwise_data,
            self.model,
            self.generator_tokenizer,
            self.generation_prompt
        )

        # 4. 保存到 JSON
        logger.info(f"Saving embeddings to {save_path} ...")

        json.dump(list_data_dict, open(save_path, "w"))
        logger.info(">>> [PeMoE] Embeddings saved.")

        return list_data_dict

    def _get_batch_embeddings(self,
                                list_data_dict,
                                model,
                                tokenizer,
                                prompt,):
        model.eval() # 确保模型处于评估模式
        all_embeddings = []
        
        # 1. 复用 get_input_data 进行批处理 (w=10 或其他 batch size)
        # 注意：这里可以直接复用文件中的 get_input_data 函数
        for input_data in get_input_data(list_data_dict, w=16): # 适当调整 batch size
            
            # 2. 格式化 Prompt (与 _generate_pairwise_data 逻辑一致)
            input_texts = [prompt.format_map(data) for data in input_data]
            
            # 3. Tokenization
            # 注意：提取 Embedding 时 padding_side 最好处理得当。
            # 只要有 attention_mask，left/right padding 都可以处理。
            input_text_tokens = tokenizer(
                input_texts,
                padding=True,
                truncation=True, # 建议加上截断防止 OOM
                max_length=self.config.llm.max_new_token, # 或其他合适的长度
                return_tensors="pt",
            ).to("cuda:0") # 确保数据在正确的设备上 (cuda:0 等)

            # 4. 前向传播 (Forward Pass)
            # 关键点：设置 output_hidden_states=True
            outputs = model(
                input_ids=input_text_tokens["input_ids"],
                attention_mask=input_text_tokens["attention_mask"],
                output_hidden_states=True
            )

            # 5. 提取最后一层 Hidden States
            # shape: (batch_size, seq_len, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]
            # 6. 提取最后一个有效 Token 的 Embedding (EOS token 或 最后一个 word)
            # 利用 attention_mask 找到最后一个非 0 元素的索引
            # attention_mask shape: (batch_size, seq_len), 1为有效, 0为pad
            # sum(dim=1) - 1 得到最后一个有效 token 的索引
            last_token_indices = input_text_tokens["attention_mask"].sum(dim=1) - 1
            
            # 从 batch 中取出对应的 embedding
            batch_embeddings = []
            for i, token_idx in enumerate(last_token_indices):
                # 取出第 i 个样本，第 token_idx 个位置的向量
                emb = last_hidden_state[i, token_idx, :].detach().float().cpu().numpy()
                batch_embeddings.append(emb)
            
            all_embeddings.extend(batch_embeddings)

        # 将 Embedding 注入回 list_data_dict 或者单独返回
        # 这里演示如何注入回字典，方便后续聚类使用
        # 确保两个列表长度一致，然后一一对应赋值
        if len(list_data_dict) == len(all_embeddings):
            for i in range(len(list_data_dict)):
                list_data_dict[i]['embedding'] = all_embeddings[i]
        else:
            # 增加一个长度检查，防止数据对不上
            logger.error(f"Mismatch! Data length: {len(list_data_dict)}, Embeddings length: {len(all_embeddings)}")

        # 关键：Numpy array 不能直接被 json 序列化，需要先转换为 list
        data_to_save = []
        for item in list_data_dict:
            # 深拷贝一份，以免影响后续逻辑
            new_item = copy.deepcopy(item)
            if 'embedding' in new_item and isinstance(new_item['embedding'], np.ndarray):
                new_item['embedding'] = new_item['embedding'].tolist()
            data_to_save.append(new_item)

        return data_to_save

    def _partition_data_by_clustering(self, list_data_dict):
        """
        直接使用 list_data_dict 中已有的 embedding 进行聚类
        """
        # 提取 Embedding 矩阵
        embeddings = np.array([np.array(item['embedding'], dtype=np.float32) for item in list_data_dict])
        
        logger.info(f">>> [PeMoE] Running K-Means on embeddings shape: {embeddings.shape}")
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=self.num_experts, random_state=self.config.seed)
        labels = kmeans.fit_predict(embeddings)

        logger.info(labels)
        # 分配数据
        partitions = {i: [] for i in range(self.num_experts)}
        for idx, label in enumerate(labels):
            partitions[label].append(list_data_dict[idx])
            list_data_dict[idx]['cluster_label'] = label

        return partitions, embeddings.shape[1]

    def _pretrain_experts(self, partitions, selector_cfg, tokenizer, prompt):
        # data_root
        _, model_name = self.config.model.type.split("@")[0].split('/', 1)
        dataset_name, _ = self.config.data.type.split("@")
        num_comp = max(2, self.config.llm.num_completions)
        
        
        # 遍历每个分区进行训练
        for expert_id, data_samples in partitions.items():
            if len(data_samples) == 0:
                logger.warning(f"Expert {expert_id} has no data. Skipping.")
                continue

            adapter_name = f"Adapter_{expert_id}"
            logger.info(f"--- Pre-training Expert {expert_id} (Adapter: {adapter_name}) ---")
            logger.info(f"Sample count: {len(data_samples)}")
            
            self.selector_model.set_active_adapter(adapter_name)
            self.selector_model.train()

            save_path = os.path.join(
                self.data_root,
                f"rlhf_prompt_{model_name}_{dataset_name}_{adapter_name}_{num_comp}.pickle"
            )
            if os.path.exists(save_path):
                try:
                    # 加载 pickle
                    token_dataset = pickle.load(open(save_path, "rb"))
                except Exception as e:
                    logger.warning(f"Failed to load prompt: {e}. Re-generating...")
            else:
                # === 构造数据集 ===
                token_dataset = LLMDataset(
                    data_samples,
                    tokenizer,
                    prompt_input=prompt,
                    prompt_no_input=prompt,
                    output_tag='choice',
                )
                pickle.dump(token_dataset, open(save_path, "wb"))
            # === 包装数据 (关键) ===
            # FS 的 Trainer 通常需要 ClientData 包装器，或者包含 train/val/test 的 dict
            # 我们创建一个只有 train_data 的简单包装
            data_wrapper = ClientData(selector_cfg, token_dataset, None, None)

            temp_config = selector_cfg.clone()
            temp_config.defrost()
            # 强制设定较小的训练步数或轮数用于预热
            temp_config.federate.total_round_num = 10
            temp_config.dataloader.batch_size = 2
            # 如果是按 step 训练，可以在这里设置
            # temp_config.train.local_update_steps = 50 
            temp_config.freeze()

            self._monitor = Monitor(temp_config, monitored_object=self)
            self.trainer = get_trainer(model=self.selector_model,
                                   data=data_wrapper,
                                   device=self.device,
                                   config=temp_config,
                                   monitor=self._monitor)

            # 开始训练
            try:
                logger.info(f"Training Adapter: {adapter_name}...")
                sample_size, model_para_all, results = self.trainer.train()
                logger.warning(sample_size)
                logger.warning(results)
                logger.info(f"Expert {expert_id} pre-training finished.")
            except Exception as e:
                logger.error(f"Error training expert {expert_id}: {e}")

        # 3. 训练结束后，可以保存一下状态，或者切换回 default (可选)
        # self.selector_model.set_active_adapter("default")
        logger.info(">>> [PeMoE] All Experts Pre-trained.")

    def _pretrain_router(self, partitions, embeddings_shape):
        """
        利用 K-Means 的聚类结果作为 Ground Truth (标签)，
        监督训练 Router 网络，使其学会根据 Embedding 预测专家 ID。
        """
        _, model_name = self.config.model.type.split("@")[0].split('/', 1)
        dataset_name, _ = self.config.data.type.split("@")
        num_comp = max(2, self.config.llm.num_completions)
        save_path = os.path.join(
                self.data_root,
                f"router_{model_name}_{dataset_name}_{num_comp}.ckpt"
            )
        # 简单的Router
        self.router = PeMoERouter(embeddings_shape, num_experts=self.num_experts, hidden_dim=128).to(self.device)

        if os.path.exists(save_path):
            try:
                # 加载权重
                state_dict = torch.load(save_path, map_location=self.device)
                self.router.load_state_dict(state_dict)
                logger.info(">>> [PeMoE] Router weights loaded successfully.")
                return # 直接返回，跳过后续训练逻辑
            except Exception as e:
                logger.warning(f">>> [PeMoE] Failed to load router weights: {e}. Re-training...")

        # 简单的 Router (示例)，实际应根据论文结构定义
        
        logger.info("Starting Router Pre-training (Distilling K-Means Knowledge)...")

        # 1. 准备数据：从 partitions 中解包出 (Embedding, Label)
        # 我们需要将所有数据展平，构建一个大的训练集
        all_embeddings = []
        all_labels = []

        for expert_id, data_samples in partitions.items():
            for sample in data_samples:
                if 'embedding' in sample:
                    all_embeddings.append(sample['embedding'])
                    all_labels.append(expert_id) # 这里的 expert_id 就是 cluster_label

        if not all_embeddings:
            logger.warning("No data found for router training! Skipping.")
            return

        # 转换为 Tensor 并移动到 GPU
        # 注意：all_embeddings 里的元素已经是 numpy array (float32)
        X = torch.tensor(np.array(all_embeddings), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(all_labels), dtype=torch.long).to(self.device)

        # 2. 创建 DataLoader
        # 使用较大的 Batch Size (如 64 或 128)，因为 MLP 计算很快
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 3. 定义优化器和损失函数
        # 这是一个多分类问题，使用 CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.router.parameters(), lr=1e-3, weight_decay=1e-4)

        # 4. 训练循环
        self.router.train()
        epochs = 15  # 这是一个简单的分类任务，通常 10-20 轮即可快速收敛
        
        for epoch in range(epochs):
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
            
            # 简单的早停或日志记录
            # 如果准确率达到 95% 以上，其实已经学得很好了
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Router Pre-train Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2%}")

        logger.info("Router Pre-training Finished.")
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            logger.info(f">>> [PeMoE] Saving trained router to {save_path} ...")
            torch.save(self.router.state_dict(), save_path)
            logger.info(">>> [PeMoE] Router saved.")
        except Exception as e:
            logger.error(f">>> [PeMoE] Failed to save router weights: {e}")
        return save_path

