# MARBERT Parameter-Efficient Adaptation Experiments

This repository contains the Jupyter Notebook used to conduct the experiments for evaluating **parameter-efficient adaptation methods for Arabic social-media text classification using MARBERT**.

The experiments compare full model fine-tuning with several Parameter-Efficient Fine-Tuning (PEFT) strategies under the same controlled experimental protocol.

## Tasks

Two Arabic social-media classification tasks are considered.

### Hate and Offensive Language Detection

Binary classification:

* **Clean**
* **Hate/Offensive (HOF)**

### Sentiment Classification

Three-class classification:

* **Positive**
* **Negative**
* **Neutral**

## Backbone Model

All experiments use the same pretrained model:

**UBC-NLP/MARBERT**

Hugging Face model:

`UBC-NLP/MARBERT`

The exact model revision used in the experiments is:

```text
88e1fa192dd723cf0b3563500aec46209762eb22
```

Using a fixed backbone allows the comparison to focus on the adaptation strategy rather than differences between pretrained language models.

## Adaptation Methods

The following strategies are evaluated:

| Method           | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| Full Fine-Tuning | All MARBERT parameters are updated                             |
| Frozen Encoder   | MARBERT is frozen and only the classification head is trained  |
| LoRA-r8          | Low-Rank Adaptation with rank 8                                |
| LoRA-r16         | Low-Rank Adaptation with rank 16                               |
| LoRA-r32         | Low-Rank Adaptation with rank 32                               |
| IA³              | Infused Adapter by Inhibiting and Amplifying Inner Activations |
| AdaLoRA          | Adaptive Low-Rank Adaptation                                   |
| DoRA-r16         | Weight-Decomposed Low-Rank Adaptation with rank 16             |

For LoRA, the query and value projections of the self-attention layers are adapted.

The main LoRA configuration is:

```text
LoRA alpha   : 32
LoRA dropout : 0.10
Target       : Query + Value projections
Bias         : None
```

AdaLoRA starts with rank 12 and adaptively reallocates its rank budget toward a target rank of 8.

DoRA uses the same rank-16 and query/value configuration as the corresponding LoRA experiment.

## Dataset

The experiments use fixed and previously audited train, validation, and test partitions.

| Split      | Number of comments |
| ---------- | -----------------: |
| Training   |            274,057 |
| Validation |             58,726 |
| Test       |             58,727 |
| **Total**  |        **391,510** |

The same records and partitions are used for both HOF and sentiment classification.

The notebook expects the following prepared files:

```text
train.csv
validation.csv
test.csv
clean_dual_label_corpus.csv
dataset_manifest.json
```

The main dataset columns used by the experiments include:

```text
ID
Comments
hate_binary
sentiment_2
```

The notebook performs additional integrity checks before training, including verification of split sizes, valid labels, duplicate-text leakage, ID overlap, class distributions, and SHA-256 hashes.

The dataset itself is not distributed through this repository. The notebook operates on the fixed prepared partitions used in the study.

## Experimental Protocol

All methods are evaluated under the same main experimental conditions.

| Parameter                 |               Value |
| ------------------------- | ------------------: |
| Random seed               |                  42 |
| Maximum sequence length   |                 128 |
| Training micro-batch size |                  32 |
| Gradient accumulation     |                   2 |
| Effective batch size      |                  64 |
| Evaluation batch size     |                  64 |
| Maximum epochs            |                  50 |
| Early-stopping patience   |                   2 |
| Model-selection metric    | Validation Macro-F1 |
| Weight decay              |                0.01 |
| Warm-up steps             |                 500 |
| Learning-rate scheduler   |              Linear |
| Maximum gradient norm     |                 1.0 |
| Mixed precision           |                FP16 |

Learning rates are:

| Method           | Learning Rate |
| ---------------- | ------------: |
| Full Fine-Tuning |        `2e-5` |
| Frozen Encoder   |        `1e-4` |
| PEFT methods     |        `2e-4` |

The **test set is never used for model selection or early stopping**.

The checkpoint obtaining the highest validation Macro-F1 is restored before final evaluation on the fixed test set.

## Evaluation Metrics

The primary evaluation metric is:

**Macro-F1**

The notebook additionally reports:

* Accuracy
* Macro-Precision
* Macro-Recall
* Macro-F1
* Weighted-F1
* Per-class Precision
* Per-class Recall
* Per-class F1
* Class support
* Confusion matrices

Individual test predictions are also retained to support paired statistical analyses and detailed error analysis.

## Computational Efficiency

In addition to predictive performance, the notebook records computational characteristics of the different adaptation strategies, including:

* Total parameters
* Trainable parameters
* Percentage of trainable parameters
* Training wall time
* Peak allocated GPU memory
* Peak reserved GPU memory
* NVML device-memory measurements
* Average GPU power
* Peak GPU power
* Estimated GPU energy consumption
* Learned artifact size

GPU energy measurements correspond to **GPU energy only** and should not be interpreted as total workstation energy consumption.

## Reproducibility

The experimental pipeline records:

* Software versions
* CUDA and GPU information
* Dataset sizes
* Class distributions
* Random seed
* MARBERT revision
* Model configurations
* Training histories
* Best checkpoints
* Test predictions
* Classification reports
* Confusion matrices
* Efficiency measurements

A fixed random seed of:

```text
42
```

is used for Python, NumPy, PyTorch, CUDA, and the Hugging Face training pipeline.

## Software Environment

The experiments reported in the notebook were executed using:

```text
Python             3.11.15
PyTorch            2.12.0+cu130
Transformers       5.14.1
PEFT               0.19.1
Datasets           5.0.0
Accelerate         1.14.0
Safetensors        0.8.0
Hugging Face Hub   1.24.0
scikit-learn       1.9.0
SciPy              1.17.1
Pandas             3.0.3
NumPy              2.4.4
PyArrow            25.0.0
OpenPyXL           3.1.5
XlsxWriter         3.2.9
nvidia-ml-py       13.610.43
psutil             7.2.2
```

The original experiments were executed using an NVIDIA CUDA-enabled GPU.

## Running the Notebook

The complete experimental pipeline is available in:

```text
1-MARBERT_PEFT_Experiments.ipynb
```

To reproduce the experiments:

1. Download or clone this repository.
2. Install the required Python packages.
3. Place the prepared dataset files in a local directory.
4. In **Section 2 — Project Directory Structure**, set `PROJECT_ROOT` and `SOURCE_PREPARED_DATA` to the corresponding local directories.
5. Execute the notebook sequentially from the first cell to the last cell.

The notebook automatically performs environment, dataset-integrity, model-configuration, and PEFT compatibility checks before launching the main experiments.

## Experiment Outputs

For each task and adaptation method, the pipeline stores the corresponding experimental artifacts, including:

```text
experiment configuration
training history
checkpoints
final learned artifact
test predictions
test errors
classification report
confusion matrix
qualitative error candidates
final result metadata
```

Completed experiments are identified through their final `result.json` file.

The notebook also supports recovery from interrupted runs by resuming from valid Hugging Face checkpoints when available.

## Notebook

**Main experimental notebook:**

`1-MARBERT_PEFT_Experiments.ipynb`

This notebook contains the complete implementation of the experimental pipeline, including dataset verification, tokenization, model construction, PEFT configuration, training, evaluation, efficiency measurement, result export, and statistical analysis.

## Citation

If you use this implementation or experimental protocol in academic work, please cite the associated paper.

The complete bibliographic information will be added following publication.
