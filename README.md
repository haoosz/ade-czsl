# Attention as Disentangler (CVPR 2023)

[![Project Page](https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white)](https://haoosz.github.io/ade-czsl/)
[![arXiv](https://img.shields.io/badge/arXiv-2303.15111%20-b31b1b)](https://arxiv.org/abs/2303.15111)
![License](https://img.shields.io/github/license/haoosz/ade-czsl?color=lightgray)

This is the official PyTorch codes for the paper:  

[**Learning Attention as Disentangler for Compositional Zero-shot Learning**](https://arxiv.org/abs/2303.15111)  
[Shaozhe Hao](https://haoosz.github.io/),
[Kai Han](https://www.kaihan.org/), 
[Kwan-Yee K. Wong](http://i.cs.hku.hk/~kykwong/)  
CVPR 2023  

### [Project Page](https://haoosz.github.io/ade-czsl/) | [Code](https://github.com/haoosz/ade-czsl) | [Paper](https://arxiv.org/abs/2303.15111)

<p align="left">
    <img src='img/teaser.gif' width="80%">
</p>

*TL;DR: A simple cross-attention mechanism is efficient to disentangle visual concepts, i.e., attribute and object concepts, enhancing CZSL performance.*

---

## Setup
Our work is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.11.0

Create a conda environment `ade` using
```
conda env create -f environment.yml
conda activate ade
```

## Download

**Datasets**: We include a script to download all datasets used in our paper. You need to download any dataset before training the model. Please download datasets from: [Clothing16K](https://drive.google.com/drive/folders/1ky5BvTFrMkPBdAWixHFGLdcfJHfu5e9_?usp=share_link) and [Vaw-CZSL](https://drive.google.com/drive/folders/1CalwDXkkGALxz0e-aCFg9xBmf7Pu4eXL?usp=sharing). You can download other datasets using
```
bash utils/download_data.sh
```
In our paper, we conduct experiments on Clothing16K, UT-Zappos50K, and
C-GQA. In the supplementary material, we also add experiments on Vaw-CZSL.

**Pretrained models**:
DINO pretrained ViT-B-16 can be found [here](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth).
We also provide models pretrained on different datasets under closed-world or open-world settings. Please download the [pretrained models](https://drive.google.com/drive/folders/1s2Ppr2bj8gDwAHBQAz33HbVzmrmbkSAI?usp=sharing) and quickly start by testing their performance using
```
python test.py --log ckpt/MODEL_FOLDER
```

## 🚀 Run the codes
### Training
Train ADE model with a specified configure file `CONFIG_FILE` (e.g, `configs/clothing.yml`) using
```
python train.py --config CONFIG_FILE
```
After training, the `logs` folder should be created with logs, configs, and checkpoints saved.

### Inference
Test ADE model with a specified log folder `LOG_FOLDER` (e.g, `logs/ade_cw/clothing`) using
```
python test.py --log LOG_FOLDER
```

## 🔥 Retrieval
### Image To Text
Conduct image-to-text retrieval with a specified log folder `LOG_FOLDER` (e.g, `logs/ade_ow/cgqa`) and a sample number `SAMPLE_NUM` (default=100).
```
python img2txt_retrieval.py --log LOG_FOLDER --sample_num SAMPLE_NUM
```

### Text To Image
Conduct text-to-image retrieval with a specified log folder `LOG_FOLDER` (e.g, `logs/ade_ow/cgqa`) and a given text prompt `TEXT` (e.g., *squatting catcher*).
```
python txt2img_retrieval.py --log LOG_FOLDER --text_prompt TEXT
```

### Visual Concept Retrieval
Conduct visual concept retrieval with a specified log folder `LOG_FOLDER` (e.g, `logs/ade_cw/clothing`) and a given image path `IMG_PATH` (e.g., `clothing/images/green_suit/000002.jpg`).
```
python visual_concept_retrieval.py --log LOG_FOLDER --image_path IMG_PATH
```

You can also adjust the coefficient $\beta$ (`args.aow`) for different datasets in retrieval tasks, referring to the chosen $\beta$ in inference:
| **Dataset** | **Closed-World** | **Open-World** |
|---------------|------------|---------------|
| Clothing | $\beta$ = 0.1 | $\beta$ = 0.1 |
| UT-Zappos | $\beta$ = 0.9 | $\beta$ = 0.9 |
| CGQA | $\beta$ = 1.0 | $\beta$ = 0.7 |

## Results
### Quantitative results

| **Dataset** | **AUC<sup>cw</sup>** | **HM<sup>cw</sup>** | **AUC<sup>ow</sup>**| **HM<sup>ow</sup>** |
|---------------|------------|---------------|------------|---------------|
| Clothing | 92.4 | 88.7 | 68.0 | 74.2 |
| UT-Zappos | 35.1 | 51.1 | 27.1 | 44.8 |
| CGQA | 5.2 | 18.0 | 1.42 | 7.6 |


### Qualitative results

From text 💬 to image 🌄: 
![image](img/txt2img.png)

From image 🌄 to text 💬:  
![image](img/img2txt.png)

## Citation
If you use this code in your research, please consider citing our paper:
```bibtex
@InProceedings{hao2023ade,
               title={Learning Attention as Disentangler for Compositional Zero-shot Learning},
               author={Hao, Shaozhe and Han, Kai and Wong, Kwan-Yee K.},
               booktitle={CVPR},
               year={2023}}
```

## Acknowledgements
Our project is based on [CZSL](https://github.com/ExplainableML/czsl). Thanks for open source!
