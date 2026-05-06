"""
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

import copy
import logging
import pandas as pd

from enum import Enum
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


# TODO: support LDA when 'category' in keys
class LLMDataset(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"], # 将原始数据中的instruction和input拼接成模型能理解的Prompt格式
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 output_tag='output'):
        super(LLMDataset, self).__init__()

        # Print prompt info
        logger.info(f'prompt_input: {prompt_input}')
        logger.info(f'prompt_no_input: {prompt_no_input}')
        # 构建 Sources（输入部分）
        self.sources = []
        for example in list_data_dict:
            input = example.get("input", None)
            if input is not None and input != "":
                self.sources.append(prompt_input.format_map(example))
            else:
                self.sources.append(prompt_no_input.format_map(example))
        # 构建 Targets（输出部分） 训练时必须告诉模型“这句话到这里结束”，否则模型推理时会停不下来
        targets = [
            f"{example[output_tag]}{tokenizer.eos_token}"
            for example in list_data_dict
        ]
        # self.llm_generated_targets = [
        #     f"{example['llm_output']}"
        #     if example.get("llm_output", "") != "" else None
        #     for example in list_data_dict
        # ]

        data_dict = self.preprocess(self.sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        self.tokenizer = tokenizer

        # 利用Pandas将文本类别转换为整数编码（codes）。例如 ["math", "code", "math"] -> [0, 1, 0]
        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest", # 按当前批次最长文本进行补零（节省显存）
                max_length=tokenizer.model_max_length,
                truncation=True, # 超长截断
            ) for text in strings
        ]
        # 提取 input_ids。因为 tokenizer 返回的是一个 batch（大小为1），所以取 [0]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        # 我们需要知道一句话到底有多长（不含补零），以便后续确定 Prompt 的边界
        input_ids_lens = labels_lens = [
            # ne(tokenizer.pad_token_id)：判断哪些位置不是 PAD Token；sum()：计算非 PAD Token 的数量
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer):
        examples = [s + t for s, t in zip(sources, targets)]
        # examples_tokenized：完整句子的分词结果 sources_tokenized：仅仅是Prompt部分的分词结果
        # 目的：我们不能简单地用字符串长度切分，因为Token的长度和字符长度不对应。必须对Source单独分词，才能获得Source对应的Token数量
        examples_tokenized, sources_tokenized = [
            self._tokenize_fn(strings, tokenizer)
            for strings in (examples, sources)
        ]
        # 模型的输入，是完整的 Prompt + Answer
        input_ids = examples_tokenized["input_ids"]
        # 初始状态下，标签=输入。这就是自回归（Auto-Regressive）训练的基础（预测下一个词）。
        labels = copy.deepcopy(input_ids)
        # 遍历每个样本的 Label 和其对应的 Prompt 长度 (source_len)
        for label, source_len in zip(labels,
                                     sources_tokenized["input_ids_lens"]):
            # 将Label中属于Prompt的那一部分切片，赋值为-100
            label[:source_len] = DefaultToken.IGNORE_INDEX.value
            # TODO: remove the data which is longer than the max input length
        # Input: [Prompt Tokens, Answer Tokens] Label: [-100, -100, ..., Answer Tokens]
        # 这样，计算 Loss 时，Prompt 部分的梯度为 0，模型只学习如何生成 Answer
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])

    # def overwrite_by_llm(self, i):
    #     source = self.sources[i]
    #     llm_answer = self.llm_generated_targets[i]

    #     if llm_answer is None or llm_answer == "":
    #         return

    #     llm_result_tknz = \
    #         self.preprocess([source], [llm_answer], self.tokenizer)
    #     self.input_ids[i], self.labels[i] = \
    #         llm_result_tknz['input_ids'][0], llm_result_tknz['labels'][0]


class LLMComparisonDataset(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 output_A='output_A',
                 output_B='output_B',
                 choice='choice'):
        new_list_data_dict = []
        for example in list_data_dict:
            if choice in example and int(example[choice]) == 1:
                # output_B is better than output_A
                example[output_A], example[output_B] = \
                    example[output_B], example[output_A]
                new_list_data_dict.append(example)
        # remove the data without choice
        list_data_dict = new_list_data_dict

        # After switching, output_A > output_B
        self.win_dataset = LLMDataset(list_data_dict=list_data_dict,
                                      tokenizer=tokenizer,
                                      prompt_input=prompt_input,
                                      prompt_no_input=prompt_no_input,
                                      output_tag=output_A)
        self.lose_dataset = LLMDataset(list_data_dict=list_data_dict,
                                       tokenizer=tokenizer,
                                       prompt_input=prompt_input,
                                       prompt_no_input=prompt_no_input,
                                       output_tag=output_B)

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

        # super(LLMComparisonDataset, self).__init__(
        #     list_data_dict, tokenizer, prompt_input,
        #     prompt_no_input, output_A)

        # self.win_labels = self.labels

        # targets_B = [
        #     f"{example[output_B]}{tokenizer.eos_token}"
        #     for example in list_data_dict
        # ]
        # data_dict_B = self.preprocess(self.sources, targets_B, tokenizer)
        # self.lose_labels = data_dict_B["labels"]

    def __len__(self):
        return len(self.win_dataset)

    def __getitem__(self, i):
        return dict(win_data=self.win_dataset[i],
                    lose_data=self.lose_dataset[i],
                    categories=self.categories[i])
