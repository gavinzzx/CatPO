# Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs.

This repository will contain the official implementation of our paper.

⛳️ Progress Update: We have almost completed all preparations. We will release the project no later than June 30 AoE.


## Brief Introduction
This repository provides the code and data for Cat-PO. Cat-PO is a novel multimodal preference alignment framework that performs fine-grained optimization of MLLM generation behavior through cross-modal adaptive token-level rewards, thereby mitigating multimodal hallucinations and improving the truthfulness of model outputs.

We fully leverage the intrinsic cross-modal capabilities of MLLMs to compute the global, local, and semantic relevance between response tokens and visual content based on cross-modal attention, image-patch attention entropy, and cross-modal semantic similarity, respectively. These relevance signals are then integrated to construct smooth token-level rewards. Benefiting from this fine-grained reward mechanism, Cat-PO can more precisely reinforce visually critical tokens and suppress hallucinated tokens without relying on additional visual detection models, APIs or external tools. Moreover, We conduct experiments on the RLHF-V dataset and validate the effectiveness of Cat-PO across multiple backbone models.

Visit our 🏠 github page and 📃 paper to learn more! We also welcome you to use our work for evaluation and research.


## Dataset and Model
Dataset: We conducted experiments using the publicly available [RLHF-V Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset).

Model: We mainly conduct our experiments in The [LLaVA-v1.5](https://github.com/haotian-liu/llava) series.

## Train
```bibtex
run catpo.sh 
```
## Evaluation
We conduct the evaluation in the general and opensource benchamrk, such as [AMBER Bench](https://github.com/junyangwang0410/AMBER), [MM-Hal Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench). 

Using the official guide for configuration and testing.




## Citation

If you find Cat-PO useful for your research, please consider cite our papers 📝 and star us ⭐️！
```bibtex
@inproceedings{zheng2026catpo,
  title     = {Cat-{PO}: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal {LLM}s},
  author    = {Zhixiao Zheng and Zheren Fu and Zhiyuan Yao and Dongming Zhang and Zhendong Mao},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=iIbe6qDN0A}
}
```

## Acknowledgement
This repo is based on [SeVa](https://github.com/Kevinz-code/SeVa) and [RLHF-V](https://github.com/RLHF-V/RLHF-V). We thank their efforts in building their codebase.


