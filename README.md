## Installation

Before you run the code, set up your environment and install FS-LLM (which is integrated into this repository). Follow these steps:

1. **Create a Conda Environment:**
   ```bash
   conda create -n fs-llm python=3.9
   conda activate fs-llm
   ```

2. **Install PyTorch (>=1.13.0):**
   ```bash
   conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

3. **Install FS-LLM in Editable Mode:**
   ```bash
   pip install -e .[llm]
   ```


---

## Attention

It should be noted that this codebase contains most of the core code of Fedprism, and all files will be added after related work is published.

## Training
1. **Run the Training Script:**  
   ```bash
   CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python federatedscope/llm/moe/main.py \
   --selector-cfg-file fedprism_script/tldr/tldr_choice_gemma_fedprism_u3.yaml \
   --cfg fedprism_script/tldr/tldr_rlhf_fedprism_gemma.yaml llm.accelerator.use True \
   federate.save_to checkpoints/tldr_rlhf_gemma__tldr_choice_gemma_fedprism_u3.ckpt \
   expname tldr/rlhf_gemma/tldr_choice_gemma_fedprism_u3
   ```

