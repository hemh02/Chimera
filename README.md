# 🔥United We Stand: Towards End-to-End Log-based Fault Diagnosis
via Interactive Multi-Task Learning

<img src="imgs/pipeline.jpg" alt="drawing" width="100%"/>

- [Chimera](#Chimera)
  
  - [Project Structure](#project-structure)
  - [Datasets](#datasets)
  - [Environment](#environment)
  + [Preparation](#preparation)
  - [Quick Start](#quick-start)

## 📌 Description

Log-based fault diagnosis is essential for maintaining software system availability. However, existing fault diagnosis methods are built using a task-independent manner, which fails to bridge the gap between anomaly detection and root cause localization in terms of data form and diagnostic objectives, resulting in three major issues: 1) Diagnostic bias accumulates in the system; 2) System deployment relies on expensive monitoring data; 3) The collaborative relationship between diagnostic tasks is overlooked. Facing this problems, we propose a novel end-to-end log-based fault diagnosis method, Chimera, whose key idea is to achieve end-to-end fault diagnosis through bidirectional interaction and knowledge transfer between anomaly detection and root cause localization. Chimera is based on interactive multi-task learning, carefully designing interaction strategies between anomaly detection and root cause localization at the data, feature, and diagnostic result levels, thereby achieving both sub-tasks interactively within a unified end-to-end framework. Evaluation on two public datasets and one industrial dataset shows that Chimera outperforms existing methods in both anomaly detection and root cause localization, achieving improvements of over 2.92%~5.00% and 19.01%~37.09%, respectively. It has been successfully deployed in production, serving an industrial cloud platform.

## 🔍 Project Structure

```
├─checkpoint      # Saved models
├─data            # Log data
├─glove           # Pre-trained Language Models for Log Embedding
├─src             
|  ├─dataset.py   # Load dataset
|  ├─models.py    # Chimera Models   
|  └─utils.py     # Log Embedding
├─main.py         # entries
└─process.py      # Data preprocess
```

## 📑 Datasets

We used `2` open-source log datasets for evaluation, BGL and Thunderbird. 

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|       ---       |           ----                     |    ----    |    ----    |  ----     |                ---                                        |
| BGL             | Blue Gene/L supercomputer log | 214.7 days | 4,747,963   | 708.76MB   | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |
| Thunderbird             | Thunderbird supercomputer log      | 244 days | 211,212,192  | 27.367  GB | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |
|                 |                               |            |             |            |                                                           |

**Note:** Considering the huge scale of the Thunderbird dataset, we followed the settings of the previous study [LogADEmpirical](https://github.com/LogIntelligence/LogADEmpirical) and selected the earliest 10 million log messages from the Thunderbird dataset for experimentation. 


## ⚙️ Environment

**Key Packages:**

Numpy==1.20.3

Pandas==1.3.5

Pytorch_lightning==1.1.2

scikit_learn==0.24.2

torch==1.13.1+cu116

tqdm==4.62.3

[Drain3](https://github.com/IBM/Drain3)


## 📜 Preparation

You need to follow these steps to **completely** run `Chimera`.
- **Step 1:** Download [Log Data](#datasets) and put it under `data` folder.
- **Step 2:** Using [Drain](https://github.com/IBM/Drain3) to parse the unstructed logs.
- **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), and put it under `glove` folder.


## 🚀 Quick Start
you can run `Chimera` on Spirit dataset with this code:

#### 👉 Stage 1: Preparing the Environment.

```
pip install -r requirement.txt
```

#### 👉 Stage 1: Training Chimera.

```
python main.py --mode train --epochs 150 --dataset BGL
```

#### 👉 Stage 2: Evaluation Chimera on BGL Dataset.

```
python main.py --mode eval --load_checkpoint True  --dataset BGL
```