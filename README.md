# MPCNN-Sleep-Apnea
We propose MPCNN, a novel feature extraction method for single-lead ECG analysis in apnea classification. Specifically, our approach is inspired by Matrix Profile (MP) algorithms, which utilize fixed-length subsequence distance profiles to capture critical features in the PQRST segment of an ECG signal. We extracted MinDP, MaxDP, and MeanDP values from these distance profiles to serve as inputs for CNN models. We compared this new feature extraction approach with conventional methods, such as R-peaks and RR intervals, in various experiments. Our results demonstrate that our technique has significant potential and efficacy for SA classification, delivering promising per-segment and per-recording performance metrics.
![First_draft_overall](https://github.com/vinuni-vishc/MPCNN-Sleep-Apnea/assets/104493696/b3e5b8b4-562e-4e98-b4e1-05911aa48411)
The overall of our proposed SA detecion. (1) Reducing noise and artifacts. (2) Our main contribution in this paper, where we generate a series of subsequences $T_{P_i,m}$, start at a P Peak and spanning a window of length $m$. Subsequently, we calculate the Euclidean distance $d_{i,j}$ to compile a distance profile $D$, from which we extract critical values: $X_{min}$ (MinDP), $X_{max}$ (MaxDP), and $X_{mean}$ (MeanDP). These values serve as inputs for the subsequent modeling stage. (3) Data split in two categories (4) We use some lightweight models to perform the result in per-segment and per-recording classification.
# Preparations
## Dataset
The data is available at https://physionet.org/content/apnea-ecg/1.0.0/ [Apnea-ECG Database]([URL](https://physionet.org/content/apnea-ecg/1.0.0/)https://physionet.org/content/apnea-ecg/1.0.0/). Please download and extract the file to `dataset`.
## Downloading dependencies
```
Python 3.10.12
Keras 2.12.0
TensorFlow 2.12.0
```
# Preprocessing
Please run the file `preprocessing.py` to extract the data for CNN's input. In our method, we also design ablation study with show as below: 
![window_size](https://github.com/vinuni-vishc/MPCNN-Sleep-Apnea/assets/104493696/905e1d8d-ae24-4cbb-ae3e-0aa09e813465)
*The design of the second ablation experiments with different window size. \\ $T_1$: From P peak to the end of ST segment. \\$T2$: From P peak to the end of QRS segment. \\$T3$: Q peaks to the end of ST segment. \\$T4$: QRS segment.*
