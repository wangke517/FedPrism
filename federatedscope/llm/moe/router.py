import torch
import torch.nn as nn
import torch.nn.functional as F

class PeMoERouter(nn.Module):
    """
    轻量级 MLP 门控网络
    结构: Input -> Linear(降维) -> Tanh/ReLU -> Linear -> Output(Logits)
    """
    def __init__(self, input_dim, num_experts, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        
        self.activation = nn.Tanh() 
        
        self.layer2 = nn.Linear(hidden_dim, num_experts)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: [batch_size, input_dim] (Prompt Embedding)
        return: logits [batch_size, num_experts]
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.layer2(x)
        return logits

    def get_routing_weights(self, x, top_k=1):
        """
        推理时使用：获取 Top-K 专家的索引和权重
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices