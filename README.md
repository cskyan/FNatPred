# FNatPred: Ensemble Model Prediction Based on Fungal Microbiome
FNatPred is a tool that can ensemble prediction information from multiple machine learning models on tumor fungal data to detecte NAT. It can also be used to predict other binary classification tasks, such as individual cancer types versus each other, Stage I vs. Stage IV, Primary Tumor vs. Solid Tissue Normal.
# Package requirement
*   python\=\=3.7
*   pandas\=\=1.3.5
*   xgboost\=\=1.5.1
*   lightgbm\=\=3.1.1
*   catboost\=\=1.1.1
*   matplotlib\=\=3.4.3
*   scikit-learn\=\=1.0.2
*   numpy\=\=1.21.5
*   scipy==1.7.3
# Installation
On the top right of the main repository page, click the button followed by the button to download the entire repository. ```Code ▼ ``` ```Download ZIP```
# Organization of files
### All of the input data used for the scripts in "code" are provided in the folder ```./data```
### All of the output data in the floder ```./output```
### The scripts in the “code” folder have the following brief descriptions
#### 1)Tumor vs. Nat
```Breast_TumorvsNAT_cv.py``` -> Five_Fold-Average-Results of our model **FNatPred** and other models in the breast tumor data.
```Breast_TumorvsNAT_multi_model_plot.py``` -> Visualization of our model **FNatPred** and other models for NAT Detection in the breast tumor data.  
```Lung_TumorvsNAT_cv.py``` -> Five_Fold-Average-Results of our model **FNatPred** and other models in the lung tumor data.
```Lung_TumorvsNAT_muti_model_plot.py``` -> Visualization of our model **FNatPred** and other models for NAT Detection in the lung tumor data.
#### 2）individual cancer types versus each other
```1vsALL_pancancers.py``` -> Comparison of Results between our model **FNatPred** and other models on Multiple Tumor Types data in the Task
#### 3）Stage I vs. Stage IV
```IvsIV_pancancers.py``` -> Comparison of Results between our model **FNatPred** and other models on Multiple Tumor Types data in the Task
#### 4）Primary Tumor vs. Solid Tissue Normal
```TumorvsNormal_pancancers.py``` -> Comparison of Results between our model **FNatPred** and other models on Partial Tumor Type data   
```TumorvsNormal_combine.py``` -> Execute after ```TumorvsNormal_pancancers.py``` to obtain more tumor type data
# Model training
To train FNatPred, two files are required as input including fungal features (e.g. species，gene, etc.) and diseases labels.
1) fungal features (required)
   
   | SampleID | Species_1 | Species_2 | ... | Species_m |  
   | ---      | ---       |    ---    | --- | ---       |  
   | Sample_1 |     0     |    2467   | ... |    7694   |
   | Sample_2 |   25600   |     0     | ... |      0    |
   | ...      |   ...   |    ...    | ... |      ...    |
   | Sample_n |  0      |    0    | ... |     0   |
   
3) Tumor labels (required)

   | SampleID | tumor_1  |   NAT  |
   | ---      | ---       |    ---    |
   | Sample_1 |     0     |    1   |
   | Sample_2 |  1        |     0    | 
   | ...      |   ...   |    ...    |
   | Sample_n |  1      |    0    |
   
# Classfication
Then you can run the```. py``` files under each task
# Contact
All problems please contact   
Dongmei He  Email：hedongmei@hainanu.edu.cn  
Shankai Yan Email: skyan@hainanu.edu.cn
