# Weakly-Supervised Video Anomaly Detection with Snippet Anomalous Attention

**Official implementation of [Paper](https://arxiv.org/abs/2309.16309)  with pytorch framework.**

> Abstract—With a focus on abnormal events contained within untrimmed videos, there is increasing interest among researchers in video anomaly detection. Among different video anomaly detection scenarios, weakly-supervised video anomaly detection poses a significant challenge as it lacks frame-wise labels during the training stage, only relying on video-level labels as coarse supervision. Previous methods have made attempts to either learn discriminative features in an end-to-end manner or employ a two-stage self-training strategy to generate snippet-level pseudo labels. However, both approaches have certain limitations. The former tends to overlook informative features at the snippet level, while the latter can be susceptible to noises. In this paper, we propose an Anomalous Attention mechanism for weakly-supervised anomaly detection to tackle the aforementioned problems. Our approach takes into account snippet-level encoded features without the supervision of pseudo labels. Specifically, our approach first generates snippet-level anomalous attention and then feeds it together with original anomaly scores into a Multi-branch Supervision Module. The module learns different areas of the video, including areas that are challenging to detect, and also assists the attention optimization. Experiments on benchmark datasets XDViolence and UCF-Crime verify the effectiveness of our method. Besides, thanks to the proposed snippet-level attention, we obtain a more precise anomaly localization.

* * *
## 1. For Mindspore Version

* **MindSpore**

> [MindSpore](https://github.com/mindspore-ai/mindspore) is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. MindSpore is designed to provide development experience with friendly design and efficient execution for the data scientists and algorithmic engineers, native support for Ascend AI processor, and software hardware co-optimization. At the meantime MindSpore as a global AI open source community, aims to further advance the development and enrichment of the AI software/hardware application ecosystem.

## 2. Usage
### 2.1 Dependencies

* Python ≥ 3.8
* MindSpore 2.0.0
* numpy
* Pillow
* scipy

### 2.2 Prepare data

The datasets used in the paper are available at the following links:

* [UCF-Crime](https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengwu%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2FUCF%2DCrime%2FI3D&ga=1)

* [XD-Violence](https://roc-ng.github.io/XD-Violence/)


* * *
## 3. Evalution
```python
python test_10crop.py
```
## 4. Result

**Result on UCF**

<img src="https://github.com/YukiFan/vad-weakly/blob/main/ucf.png" width="50%">

**Result on XD-violence**

<img src="https://github.com/YukiFan/vad-weakly/blob/main/xd.png" width="50%">

## Citation
If you find our code or paper useful, please consider citing:

```
@article{fan2023weakly,
  title={Weakly-Supervised Video Anomaly Detection with Snippet Anomalous Attention},
  author={Fan, Yidan and Yu, Yongxin and Lu, Wenhuan and Han, Yahong},
  journal={arXiv preprint arXiv:2309.16309},
  year={2023}
}
```


