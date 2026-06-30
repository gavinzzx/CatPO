# Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs.

This repository will contain the official implementation of our paper.


## Brief Introduction
This repository provides the code and data for Cat-PO, which is a novel multimodal preference optimization framework based on cross-modal adaptive token-level rewards, thereby mitigating hallucinations and improving the truthfulness. The Cat-PO framework is shown below.

<p align="center">
  <img src="image/Cat-PO-frame.png" width="850">
</p>

<p align="center">
  Overview of the proposed Cat-PO framework.
</p>

We fully leverage the intrinsic cross-modal capabilities of MLLMs to compute the global, local, and semantic relevance between response tokens and visual content based on cross-modal attention, patch entropy, and semantic similarity, respectively. These relevance signals are then integrated to construct token-level rewards. Benefiting from this fine-grained reward mechanism, Cat-PO can more precisely reinforce visually critical tokens and suppress hallucinated tokens without relying on external tools or APIs. We conduct the experiments and validate the effectiveness of Cat-PO.

Visit our 🏠[github page](https://github.com/gavinzzx/CatPO) and 📝[paper](https://openreview.net/pdf?id=iIbe6qDN0A) to learn more! We also welcome you to use our work for research and evaluation.


## Dataset and Model
Dataset: We conducted experiments using the publicly available [RLHF-V Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset).

Model: We mainly conduct our experiments in The [LLaVA-v1.5](https://github.com/haotian-liu/llava) series.

## Train
Most of the configurations are already complete in catpo.sh, let's start quickly from the following commands.

```bibtex
bash catpo.sh 
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


