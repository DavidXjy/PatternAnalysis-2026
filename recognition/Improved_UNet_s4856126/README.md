# Improved 2D U-Net for HipMRI Segmentation

**Course:** COMP3710 Pattern Recognition  
**Student ID:** s4856126  
**Difficulty:** Normal  
**Framework:** PyTorch  
**Dataset:** HipMRI Study

## 1. Introduction

Accurate medical image segmentation is important for radiotherapy planning, where clinicians must distinguish target structures from surrounding healthy tissue. In pelvic MRI, anatomical boundaries may be difficult to identify because of low soft-tissue contrast, variation between patients, and small target structures.

This project investigates whether an Improved 2D U-Net with residual convolutional blocks can provide more accurate anatomical segmentation than a Standard 2D U-Net.

The central research question is whether additional architectural complexity improves segmentation accuracy, particularly around small structures and challenging boundaries, sufficiently to justify its computational cost.

A Standard U-Net is implemented as the baseline. Its performance is compared with a Residual U-Net using the same dataset partitions, training duration, optimizer, learning rate, and evaluation methodology.

The project is intended as an experimental research prototype rather than a clinically validated radiotherapy planning system.

## 2. Dataset and Preprocessing

### 2.1 Dataset

The project uses the HipMRI Study dataset provided on the University of Queensland Rangpur HPC cluster.

The 2D dataset is located at:

`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/`

Each sample consists of an MRI slice and a corresponding segmentation mask stored in NIfTI format (`.nii.gz`).

The segmentation masks contain six numerical labels, from 0 to 5. These represent the background and annotated anatomical structures.

The official numerical-to-anatomical label mapping was not available in the inspected local dataset metadata. Although visual inspection suggests that raw Label 5 may correspond to the prostate, this has not been authoritatively confirmed.

Therefore, results are reported using raw label IDs until the mapping is verified.

### 2.2 Dataset Splitting

The dataset contains the following predefined partitions:

| Partition | MRI slices | Patients |
|---|---:|---:|
| Training | 11,460 | 31 |
| Validation | 660 | 4 |
| Test | 540 | 3 |

The patient groups are disjoint across all three partitions, as verified using patient identifiers extracted from image filenames.

Patient-level separation is essential because different MRI slices from the same patient may contain highly correlated anatomical information. Mixing slices from one patient across training and testing could artificially inflate reported performance.

The training partition is used to optimize model parameters, the validation partition is used for checkpoint selection, and the test partition is reserved for final evaluation.

### 2.3 Preprocessing

The data-loading and preprocessing pipeline includes:

1. Loading MRI slices and segmentation masks using NiBabel.
2. Matching MRI files with their corresponding segmentation masks.
3. Checking image-mask alignment, dimensionality, and label validity.
4. Using the 256 × 128 spatial resolution of the prepared 2D slices.
5. Applying per-slice Z-score intensity normalisation.
6. Converting MRI images into floating-point PyTorch tensors.
7. Converting segmentation masks into integer class labels.
8. Applying random intensity augmentation to the training partition only.

MRI intensity normalisation is applied without changing segmentation class identities.

Where image resampling is required for testing or preprocessing, continuous MRI intensities should use bilinear interpolation and categorical masks should use nearest-neighbour interpolation to preserve valid class labels.

The resulting input MRI tensor has shape `(1, 256, 128)`, while the corresponding multiclass mask has shape `(256, 128)`.

The network predicts six segmentation classes at the input spatial resolution.

## 3. Baseline: Standard 2D U-Net

### 3.1 Architecture

The baseline is a Standard 2D U-Net implemented in `modules.py`.

The model consists of an encoder, a bottleneck, and a decoder connected by skip connections.

The encoder progressively reduces spatial resolution while learning increasingly abstract visual features. The decoder reconstructs the segmentation map using upsampling and features transferred from corresponding encoder stages.

Skip connections preserve spatial information that might otherwise be lost during downsampling, which is especially important for segmentation boundaries.

The model uses convolutional layers, batch normalisation, ReLU activations, pooling, and upsampling operations.

It accepts a one-channel MRI image and produces six-channel segmentation logits.

### 3.2 Training Configuration

The baseline was trained on an NVIDIA A100 GPU using the following configuration:

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Input channels | 1 |
| Output classes | 6 |
| Base channels | 16 |
| Input resolution | 256 × 128 |

The model was evaluated on the validation partition after each epoch.

The checkpoint with the highest validation foreground macro Dice was saved automatically.

### 3.3 Baseline Validation Results

The Standard U-Net completed all 10 training epochs successfully.

| Metric | Result |
|---|---:|
| Best epoch | 6 |
| Best validation foreground macro Dice | 0.9109 |
| Final training loss, Epoch 10 | 0.0265 |

The per-class validation Dice scores at the selected checkpoint were:

| Raw label | Validation Dice |
|---|---:|
| 0 | 0.9950 |
| 1 | 0.9832 |
| 2 | 0.9366 |
| 3 | 0.8995 |
| 4 | 0.8915 |
| 5 | 0.8440 |

The foreground macro Dice excludes raw Label 0, the background class.

### 3.4 Baseline Training Curves

![Standard U-Net Training Curves](images/baseline_training_curves.png)

**Figure 1.** Training and validation performance of the Standard U-Net over 10 epochs.

Validation foreground macro Dice reached its highest value of 0.9109 at Epoch 6.

Although training loss continued to decrease, validation performance fluctuated after Epoch 6, suggesting possible overfitting or training variability.

The best-performing checkpoint was selected from Epoch 6 and saved as:

`checkpoints/baseline_unet.pt`

These results represent validation performance rather than independent test-set performance.

## 4. Improved 2D U-Net

### 4.1 Motivation

Although Standard U-Net provides strong baseline performance, segmenting small anatomical structures and ambiguous boundaries remains challenging.

An important limitation of deeper convolutional networks is that optimization and gradient propagation may become more difficult as network depth increases.

The proposed improvement replaces standard convolutional blocks with residual convolutional blocks while preserving the overall U-Net encoder-decoder design.

The aim is to investigate whether residual connections improve feature learning, segmentation accuracy, and boundary reliability.

### 4.2 Residual Blocks

A residual block learns a transformation of its input and adds a shortcut connection:

**y = ReLU(F(x) + S(x))**

Here:

- `F(x)` represents the transformation learned by the convolutional layers.
- `S(x)` represents the residual shortcut.
- The shortcut is an identity mapping when channel dimensions match.
- A 1 × 1 convolution is used when input and output channel dimensions differ.

The shortcut allows features and gradients to propagate through the network more directly.

Residual connections can facilitate optimization, but improved segmentation performance must still be demonstrated experimentally.

### 4.3 Improved Architecture

The Improved U-Net is implemented in `improved_modules.py`.

Its main components include:

- Residual convolutional blocks in the encoder and decoder.
- Max pooling for spatial downsampling.
- Transposed convolutions for upsampling.
- Skip connections between corresponding encoder and decoder stages.
- A final 1 × 1 convolution producing six segmentation logits.

The architecture preserves the main spatial structure of the Standard U-Net, allowing a comparison of ordinary convolutional blocks and residual convolutional blocks.

The implementation successfully passed input-output shape tests, forward propagation tests, backward propagation tests, and optimizer smoke tests.

### 4.4 Training Configuration

The Improved U-Net was trained using:

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Input channels | 1 |
| Output classes | 6 |
| Base channels | 16 |
| Input resolution | 256 × 128 |

The training and validation partitions and preprocessing pipeline were identical to those used for Standard U-Net.

The primary experimental difference was the model architecture.

### 4.5 Improved U-Net Validation Results

The Improved U-Net completed all 10 training epochs successfully on an NVIDIA A100 GPU.

The Slurm training job completed in 16 minutes and 7 seconds.

| Metric | Result |
|---|---:|
| Best epoch | 10 |
| Best validation foreground macro Dice | 0.9039 |
| Final training loss, Epoch 10 | 0.0239 |

Per-class validation Dice at Epoch 10:

| Raw label | Validation Dice |
|---|---:|
| 0 | 0.9949 |
| 1 | 0.9829 |
| 2 | 0.9362 |
| 3 | 0.8773 |
| 4 | 0.8867 |
| 5 | 0.8365 |

The best checkpoint was saved as:

`checkpoints/improved_unet.pt`

The Improved U-Net achieved a lower final training loss than Standard U-Net, but its best validation foreground macro Dice was slightly lower.

This indicates that adding residual blocks did not produce a clear validation accuracy improvement in the current 10-epoch experiment.

However, one training run per architecture is insufficient to establish whether this difference is statistically reliable.

### 4.6 Improved U-Net Training Curves

![Improved U-Net Training Curves](images/improved_training_curves.png)

**Figure 2.** Training and validation performance of the Improved U-Net over 10 epochs.

The Improved U-Net achieved its highest validation foreground macro Dice of 0.9039 at Epoch 10.

Although training loss decreased steadily, validation Dice fluctuated throughout training. Compared with Standard U-Net, the improved architecture achieved lower final training loss but slightly lower best validation Dice.

These findings suggest that residual connections did not produce a clear generalisation advantage under the current experimental configuration.

## 5. Feasibility Review

**Project:** Improved 2D U-Net for HipMRI Segmentation  
**Student ID:** s4856126  
**Difficulty:** Normal

### 5.1 User Need, Scope and Acceptance Criteria

This project investigates automated contour proposals for pelvic MRI to assist medical imaging specialists in prostate radiotherapy planning.

The system is intended for expert review, not autonomous clinical use.

The acceptance criteria are:

1. Maintain independent patient-level training, validation and test splits.
2. Target prostate Dice ≥ 0.75 on the held-out test set after verifying the anatomical label mapping.
3. Compare Standard U-Net and Improved U-Net using per-class Dice and IoU.
4. Measure model parameters, peak GPU memory and inference latency.
5. Investigate 3–5 actual segmentation failures involving small targets and ambiguous boundaries.

### 5.2 Model Choice and Course Concepts

The project uses HipMRI 2D slices available on the UQ Rangpur HPC cluster.

Standard 2D U-Net serves as the baseline, while Improved 2D U-Net introduces residual convolutional blocks into the encoder-decoder architecture.

The design draws on CNN feature extraction, U-Net skip connections, residual learning, cross-entropy loss and gradient-based optimization.

The central research question is whether residual connections improve segmentation accuracy enough to justify the additional computational cost.

### 5.3 Preliminary Feasibility Evidence

An initial dataset audit confirmed disjoint patient partitions and six numerical mask labels:

| Partition | MRI slices | Patients |
|---|---:|---:|
| Training | 11,460 | 31 |
| Validation | 660 | 4 |
| Test | 540 | 3 |

The prepared 2D slices use 256 × 128 pixels, with integer mask labels preserved.

Data-loading, tensor-shape, and forward/backward smoke tests demonstrated an operational PyTorch training pipeline.

### 5.4 Risks, Computing Budget and Fallback

The main risks include class imbalance, small or ambiguous boundaries, limited patient diversity, overfitting, lack of 3D context, and uncertainty in the numerical-to-anatomical label mapping.

Experiments use NVIDIA A100 GPUs through Rangpur Slurm, with an initial configuration of 10 epochs, batch size 8 and a three-hour training-job allocation.

The original plan was to train both networks, benchmark them on held-out test data, profile computational resources and inspect difficult segmentation cases.

Standard U-Net remains the fallback if residual blocks provide insufficient benefit.

**Practical check-off status:** Tutor consultation pending. Completion must be recorded by the teaching team independently of this written review.

## 6. Experimental Comparison

### 6.1 Validation Performance

Both models were trained for 10 epochs using the same dataset partitions and principal training hyperparameters.

| Metric | Standard U-Net | Improved U-Net |
|---|---:|---:|
| Training epochs | 10 | 10 |
| Best validation epoch | 6 | 10 |
| Best validation foreground macro Dice | **0.9109** | 0.9039 |
| Label 5 validation Dice | **0.8440** | 0.8365 |
| Final training loss | 0.0265 | **0.0239** |
| Trainable parameters | 1,942,374 | 2,030,902 |

The Standard U-Net achieved approximately 0.0070 higher validation foreground macro Dice.

For raw Label 5, Standard U-Net achieved approximately 0.0075 higher Dice.

Although Improved U-Net produced lower training loss, it did not outperform the baseline on validation accuracy.

This suggests that additional residual structure did not provide a measurable validation accuracy advantage in the current experiment.

Possible explanations include differences in optimization dynamics, limited training diversity, architecture-specific inductive biases and ordinary random variation between training runs.

The independent test results and case-specific errors are analysed below.

### 6.2 Independent Test Set Evaluation

Both selected checkpoints were evaluated on the complete held-out test partition containing **540 MRI slices from three patients**.

The same `evaluate.py` script, GPU device and batch size of 8 were used for both models.

Each class Dice and Intersection-over-Union (IoU) was computed by accumulating pixel-level confusion counts across the full test partition, rather than averaging per-slice Dice scores.

#### Per-class Dice and IoU

| Raw label | Standard Dice | Improved Dice | Standard IoU | Improved IoU |
|---|---:|---:|---:|---:|
| 0 | 0.9969 | 0.9970 | 0.9939 | 0.9940 |
| 1 | 0.9844 | 0.9845 | 0.9693 | 0.9694 |
| 2 | 0.9353 | 0.9353 | 0.8784 | 0.8785 |
| 3 | **0.9654** | 0.9640 | **0.9331** | 0.9304 |
| 4 | **0.8424** | 0.8358 | **0.7277** | 0.7180 |
| 5 | **0.8637** | 0.8511 | **0.7600** | 0.7408 |

#### Overall Performance and Resource Comparison

| Metric | Standard U-Net | Improved U-Net |
|---|---:|---:|
| Test mean foreground Dice | **0.9182** | 0.9141 |
| Test Label 5 Dice | **0.8637** | 0.8511 |
| Test Label 5 IoU | **0.7600** | 0.7408 |
| Inference time per image | 2.455 ms | 1.376 ms |
| Peak allocated GPU memory | **149.47 MB** | 165.82 MB |
| Trainable parameters | **1,942,374** | 2,030,902 |

On the independent test set, Standard U-Net achieved a mean foreground Dice of **0.9182**, compared with **0.9141** for Improved U-Net.

The difference was approximately **0.0041** in favour of the baseline.

For raw Label 5, Standard U-Net achieved Dice **0.8637** and IoU **0.7600**, while Improved U-Net achieved Dice **0.8511** and IoU **0.7408**.

This represents a Dice advantage of approximately **0.0126** for Standard U-Net on raw Label 5.

Both models achieved raw Label 5 Dice above 0.75. However, this result must not be described as confirmed prostate Dice until the numerical-to-anatomical label mapping is verified.

The official target of Dice ≥ 0.75 refers specifically to the prostate label.

The test metric aggregation differs from the reporting procedure used in some training experiments. Therefore, comparisons between architectures use results produced by the same complete test evaluation script.

Only three patients are represented in the test partition. These findings should not be interpreted as evidence of general clinical performance.

The evaluation produced the following files:

- `outputs/evaluation/baseline_test_cuda.json`
- `outputs/evaluation/baseline_test_cuda.csv`
- `outputs/evaluation/improved_test_cuda.json`
- `outputs/evaluation/improved_test_cuda.csv`

### 6.3 Resource Efficiency

Computational efficiency is an important consideration for deploying segmentation models in practical annotation workflows.

Standard U-Net contains **1,942,374 trainable parameters**, while Improved U-Net contains **2,030,902 parameters**.

The residual architecture therefore adds **88,528 parameters**, representing a **4.56% increase** relative to the baseline.

The measured peak allocated GPU memory was:

- Standard U-Net: **149.47 MB**
- Improved U-Net: **165.82 MB**

This represents an approximately **10.94% increase** in peak allocated GPU memory for the residual architecture.

The recorded inference times were:

- Standard U-Net: **2.455 ms per image**
- Improved U-Net: **1.376 ms per image**

The Improved U-Net recorded a shorter inference time in this evaluation. However, these values were obtained from a single evaluation procedure rather than a repeated, warmed-up latency benchmark.

Therefore, the apparent speed advantage should be interpreted cautiously and should not be treated as evidence of consistently faster inference.

Peak allocated CUDA memory also measures allocated tensor memory rather than total physical GPU memory consumption.

Overall, Improved U-Net required more parameters and allocated GPU memory but did not achieve higher test foreground Dice.

This demonstrates that greater architectural complexity does not necessarily produce better segmentation accuracy or a more favourable engineering trade-off.

## 7. Qualitative Failure Case Analysis

### 7.1 Evaluation Method

Global Dice scores can hide important local errors, particularly in small structures and uncertain anatomical boundaries.

The `failure_cases.py` script evaluated both models on the same held-out test slices and identified five informative failure cases for raw Label 5.

For each selected image, the script generated an eight-panel comparison containing:

1. Original MRI image.
2. Ground-truth raw Label 5 mask.
3. Standard U-Net prediction overlay.
4. Improved U-Net prediction overlay.
5. Standard U-Net false-positive region.
6. Standard U-Net false-negative region.
7. Improved U-Net false-positive region.
8. Improved U-Net false-negative region.

Ground-truth contours are displayed in red and prediction contours in green.

False-positive pixels are highlighted in red, while false-negative pixels are highlighted in blue.

The investigation focuses on raw Label 5 because its precise anatomical correspondence has not yet been authoritatively verified.

### 7.2 Case-by-Case Results

The following five cases were selected from actual test-set predictions:

| Test Index | GT Label 5 Pixels | Standard Dice | Improved Dice | Main Finding |
|---|---:|---:|---:|---|
| 8 | 16 | 0.0000 | 0.0000 | Both models miss a small target |
| 73 | 42 | 0.0000 | 0.0000 | Both models miss the target |
| 130 | 5 | 0.0000 | 0.0000 | Extremely small target is missed |
| 250 | 171 | **0.8518** | 0.0568 | Improved U-Net suffers severe under-segmentation |
| 335 | 108 | 0.2581 | **0.7219** | Improved U-Net substantially improves the prediction |

#### Case 8: Small Target Missed by Both Models

**Ground-truth Label 5 pixels:** 16  
**Standard U-Net Dice:** 0.0000  
**Improved U-Net Dice:** 0.0000

The ground-truth mask contains a very small labelled region located near a prominent image structure.

Neither model produced a corresponding successful segmentation of this region.

The error maps show false-negative pixels in the ground-truth area, indicating that the annotated structure was omitted.

This is an example of a complete missed-detection failure for both architectures.

A plausible contributing factor is the extremely small target size relative to the full 256 × 128 input image.

#### Case 73: Recurring Small-Structure Failure

**Ground-truth Label 5 pixels:** 42  
**Standard U-Net Dice:** 0.0000  
**Improved U-Net Dice:** 0.0000

Both models failed to segment the annotated Label 5 region.

Although the target is larger than in Case 8, it still occupies a very small fraction of the total image.

The false-negative overlays show that neither architecture recovered the expected region.

This demonstrates that small-object detection remains a recurring weakness of both models rather than an isolated failure.

The result also suggests that simply introducing residual connections is insufficient to eliminate small-target segmentation errors.

#### Case 130: Extremely Small Target

**Ground-truth Label 5 pixels:** 5  
**Standard U-Net Dice:** 0.0000  
**Improved U-Net Dice:** 0.0000

This case contains an extremely small ground-truth region consisting of only five pixels.

Neither model produced an overlapping prediction.

This type of structure may be particularly difficult to recognise because of spatial information loss during encoder downsampling, severe class imbalance, and uncertainty around the precise annotated boundary.

These factors are plausible explanations, but the exact causal mechanism cannot be determined from one image alone.

The case highlights a limitation of slice-based segmentation when the target occupies only a tiny fraction of the input.

#### Case 250: Severe Under-Segmentation by Improved U-Net

**Ground-truth Label 5 pixels:** 171  
**Standard U-Net Dice:** 0.8518  
**Improved U-Net Dice:** 0.0568

This is one of the most important cases in the experiment.

Standard U-Net successfully identified most of the annotated target region, with its prediction boundary closely following the ground-truth contour.

In contrast, Improved U-Net predicted only a small part of the target.

Its false-negative overlay shows that a substantial portion of the annotated region was omitted.

This represents severe under-segmentation and a major decrease in local Dice.

The difference between the models is approximately **0.7949 Dice** in favour of Standard U-Net.

Importantly, this target contains 171 pixels and is not an extremely small five-pixel structure.

Therefore, the failure cannot be explained solely by the target being exceptionally small.

The case demonstrates that a more complex architecture may produce unstable local predictions even when its overall segmentation performance remains relatively high.

#### Case 335: Improved U-Net Successfully Recovers More of the Target

**Ground-truth Label 5 pixels:** 108  
**Standard U-Net Dice:** 0.2581  
**Improved U-Net Dice:** 0.7219

This case shows the opposite pattern to Case 250.

Standard U-Net identified only a limited part of the ground-truth target, resulting in a low Dice score of 0.2581.

Improved U-Net produced a segmentation that overlapped substantially more of the annotated region.

The false-negative area was reduced, and Dice increased to 0.7219.

The improvement was approximately **0.4638 Dice**.

This provides evidence that residual connections can be beneficial for particular difficult slices.

However, the improvement was not consistent across all cases and did not translate into better overall test Dice.

### 7.3 Cross-Case Interpretation

The five cases demonstrate three important findings.

**First, both models struggle with extremely small anatomical regions.**

Cases 8, 73 and 130 all produced Dice scores of zero for both architectures.

This indicates a recurring failure mode involving small or weakly represented target structures.

**Second, residual connections do not guarantee greater boundary reliability.**

Case 335 demonstrated a substantial improvement from the residual architecture, whereas Case 250 demonstrated a severe deterioration.

The same architectural modification can therefore produce very different local effects across images.

**Third, high average Dice can conceal serious segmentation failures.**

Both architectures achieved test foreground macro Dice above 0.91, yet several selected slices were segmented poorly or completely missed.

These cases were deliberately selected for difficult behaviour and are not representative of the failure frequency across the entire test partition.

They provide qualitative evidence about failure modes rather than an unbiased estimate of general model reliability.

This directly addresses the project's engineering dilemma: improving architectural complexity does not automatically ensure safer or more reliable anatomical boundaries.

### 7.4 Practical Implications and Future Improvements

For a human-in-the-loop anatomical contouring system, automatically generated masks should be treated as editable proposals rather than final clinical annotations.

Particular attention should be given to slices containing:

- Very small labelled regions.
- Missing target predictions.
- Large false-negative areas.
- Ambiguous anatomical boundaries.
- Strong disagreements between different model predictions.

Potential future improvements include class-balanced loss functions, Dice-based training objectives, stronger sampling of small targets, and architectures that incorporate information across adjacent MRI slices.

These are proposed improvements rather than additional experiments performed in this submission.

The five complete eight-panel comparison figures and the corresponding `test_case_metrics.csv` were generated from real test data.

Patient-level MRI figures are retained in private research outputs pending confirmation of public redistribution permissions. They are not included in the public GitHub repository at this stage.

## 8. Usage and Reproducibility

### 8.1 Project Structure

| File | Purpose |
|---|---|
| `dataset.py` | Data loading, preprocessing, augmentation and split verification |
| `modules.py` | Standard U-Net architecture |
| `improved_modules.py` | Improved Residual U-Net architecture |
| `train.py` | Baseline training and validation |
| `train_improved.py` | Improved-model training using the shared pipeline |
| `predict.py` | Segmentation inference and visualisation |
| `evaluate.py` | Dataset-level Dice, IoU and resource evaluation |
| `failure_cases.py` | Failure-case selection and visual analysis |
| `README.md` | Project documentation and experimental report |

### 8.2 Software Environment

The experiments were conducted using the University of Queensland Rangpur HPC environment.

The verified software versions are:

| Software | Version |
|---|---|
| Python | 3.11.13 |
| PyTorch | 2.14.1+cu130 |
| NumPy | 2.4.6 |
| NiBabel | 5.4.2 |
| Matplotlib | 3.11.2 |

GPU training and evaluation were performed using NVIDIA A100 hardware.

The original dataset is stored on Rangpur and is not included in the repository.

### 8.3 Training

The Standard U-Net can be trained using:

`python train.py --epochs 10 --batch-size 8 --device cuda --checkpoint checkpoints/baseline_unet.pt --output-dir outputs/baseline_10epochs`

The Improved U-Net can be trained using:

`python train_improved.py --epochs 10 --batch-size 8 --device cuda --checkpoint checkpoints/improved_unet.pt --output-dir outputs/improved_10epochs`

These GPU commands should be executed within an appropriately allocated Rangpur GPU job rather than directly on the login node.

The training scripts save model checkpoints and training histories.

### 8.4 Prediction

Example baseline inference:

`python predict.py --model baseline --index 0`

Example improved-model inference:

`python predict.py --model improved --index 0`

The prediction script loads the selected trained checkpoint, generates a segmentation prediction and saves a visual comparison.

### 8.5 Evaluation

A lightweight evaluation functionality check can be run using:

`python evaluate.py --model baseline --split validate --device cpu --max-samples 2`

This command evaluates only two validation slices and does not produce a representative accuracy benchmark.

Full test-set evaluation can be performed using:

`python evaluate.py --model baseline --split test --device cuda --batch-size 8`

`python evaluate.py --model improved --split test --device cuda --batch-size 8`

The evaluation script saves numerical results as CSV and JSON files.

Failure-case analysis can be performed using:

`python failure_cases.py --split test --device cuda --label 5 --top-k 5 --batch-size 8`

Full GPU evaluations should be executed through the Rangpur Slurm scheduler.

### 8.6 Reproducibility Notes

Model selection is performed using validation performance rather than test-set performance.

The same predefined data partitions are retained for both architectures.

Model checkpoints, full training data and generated outputs are stored separately from the source-code repository.

Reported numerical performance is based on actual model training and evaluation logs rather than synthetic or illustrative values.

A single experimental run does not quantify variance across random seeds. More extensive repeat experiments would be necessary to determine the statistical robustness of observed performance differences.

## 9. Engineering Recommendation

### 9.1 Overall Recommendation

Based on the completed validation and independent test-set experiments, **Standard U-Net is the recommended architecture for further development under the current experimental conditions**.

On the same held-out test partition, Standard U-Net achieved:

- Higher mean foreground Dice: **0.9182 versus 0.9141**.
- Higher raw Label 5 Dice: **0.8637 versus 0.8511**.
- Fewer parameters: **1,942,374 versus 2,030,902**.
- Lower peak allocated GPU memory: **149.47 MB versus 165.82 MB**.

The residual architecture achieved lower training loss but did not provide a global test Dice advantage.

Although Improved U-Net recorded shorter inference latency in the initial measurement, this result requires repeated, warmed-up benchmarking before a consistent speed advantage can be established.

Therefore, Standard U-Net currently provides the more favourable combination of segmentation accuracy and model footprint.

### 9.2 Error Reliability and Operational Risk

The five failure cases demonstrate that both models can completely miss very small annotated regions.

The residual architecture improved Case 335 substantially but also failed dramatically in Case 250, where Standard U-Net performed well.

These contrasting cases illustrate why average segmentation metrics alone cannot establish boundary reliability.

In radiotherapy planning, missed or incorrectly included anatomical regions may have important clinical consequences.

Consequently, automated segmentation predictions should be treated as proposals that require specialist verification.

Particular caution is necessary around small targets, missing regions and uncertain boundaries.

Additional safeguards such as uncertainty estimation, cross-slice consistency analysis and targeted manual review may improve the safety of future systems.

However, these safeguards were not implemented or validated in this experiment.

### 9.3 Limitations and Future Work

The present study has several limitations.

The held-out test partition contains only three patients, which restricts the generalisability of the results.

Each architecture was trained in a single 10-epoch experiment. Multiple random seeds would be required to assess training variability more reliably.

The models operate on independent 2D slices and do not explicitly model continuity across adjacent MRI slices.

The numerical-to-anatomical class mapping, including whether raw Label 5 represents the prostate, remains unverified.

The inference time measurements were obtained from a single evaluation procedure rather than a repeated latency benchmark.

Future work should focus on verifying anatomical labels, evaluating additional patients, experimenting with class-balanced or Dice-based loss functions, and investigating 3D or multi-slice architectures.

Repeated benchmarking of GPU memory and inference latency would also strengthen the computational comparison.

### 9.4 Final Recommendation to the Project Manager

The experimental evidence does not justify selecting the residual architecture solely because it is more complex.

Standard U-Net achieved slightly better global segmentation accuracy while requiring fewer parameters and less peak allocated GPU memory.

Improved U-Net demonstrated a meaningful advantage on one difficult slice, but this was accompanied by severe deterioration on another.

For the next stage of prototype development, Standard U-Net is recommended as the primary benchmark.

Further research should focus on reducing small-structure detection failures and improving boundary reliability rather than increasing architectural complexity without targeted evidence of benefit.

Neither model should be deployed as an autonomous clinical segmentation system without verified anatomical labels, more extensive patient-level testing and expert validation.

## 10. Artificial Intelligence Usage Disclosure

ChatGPT was used to assist with understanding PyTorch concepts, developing and debugging data-loading and model-training code, preparing evaluation and visualisation scripts, interpreting diagnostic outputs and structuring project documentation.

AI assistance included suggestions for implementing residual blocks, structuring reproducible experiments, and identifying potential problems involving medical image preprocessing, data leakage and model evaluation.

The suggested code was checked through dataset audits, patient-split verification, tensor shape tests, forward and backward propagation smoke tests, checkpoint inspection and actual training experiments on Rangpur.

Both models completed 10-epoch training runs, and the training logs confirmed that best-validation checkpoints were saved.

Independent test-set evaluation and five visual failure-case analyses were completed using actual HipMRI data.

The author remains responsible for validating the implementation, experimental claims and conclusions. Unverified anatomical mappings and experimental limitations are explicitly identified.

## 11. Data Acknowledgment and Usage Restrictions

### 11.1 Acknowledgment

Data were provided in part by the HipMRI Study.

The HipMRI Study dataset was used for non-commercial academic research as part of the COMP3710 Pattern Recognition project at the University of Queensland.

The contributions of the dataset creators and the researchers responsible for MRI acquisition and anatomical segmentation are acknowledged.

### 11.2 Data Usage Restrictions

The project follows the HipMRI Study data use agreement.

The agreement prohibits commercial use of the dataset and attempts to identify or contact the included human participants.

The dataset is used only for educational and non-commercial research purposes.

The original MRI scans, segmentation masks and trained model checkpoints are not distributed through this GitHub repository.

Public redistribution of patient-level MRI visualisations or derived overlays will only be considered after confirming the applicable permissions.

The dataset providers do not guarantee the correctness of the provided data. Therefore, preprocessing assumptions, annotation quality and anatomical label mappings must be independently checked where possible.

### 11.3 Dataset Citation

HipMRI Study for MRI-alone treatment planning and other imaging.

https://doi.org/10.1016/j.ijrobp.2015.08.045

This publication is included among the references specified in the HipMRI Study data use agreement.

## 12. References

1. The University of Queensland. *COMP3710/COMP7710 Pattern Recognition: Report Assessment Specification*. 2026.

2. HipMRI Study. *HipMRI Study for MRI-alone Treatment Planning and Other Imaging*. https://doi.org/10.1016/j.ijrobp.2015.08.045

3. Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. arXiv:1505.04597. https://arxiv.org/abs/1505.04597

4. He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. https://arxiv.org/abs/1512.03385

5. HipMRI Study. *Data Use Agreement*. Distributed with the HipMRI Study dataset on the UQ Rangpur HPC cluster.