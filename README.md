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

<img src="https://github.com/Daniel00008/WS-VAD-mindspore/blob/main/ucf.png" width="50%">

**Result on XD-violence**

<img src="https://github.com/Daniel00008/WS-VAD-mindspore/blob/main/xd.png" width="50%">




