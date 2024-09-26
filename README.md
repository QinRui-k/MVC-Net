# Multi-View Consistency Perception Network (MVC-Net)

## ✈️ Highlights
🔥 A novel semi-supervised segmentation paradigm named MVC-Net based on multi-view consistency-perception.  
🔥 MVC-Net achieves outstanding performance with a limited amount of annotated data on our self-constructed largest cranio-maxillofacial multi-tissue and tumor dataset named CMT-TS dataset and three other publicly available datasets.  
🔥 The effectiveness of MVC-Net is demonstrated from two perspectives in multi-view learning namely the multi-view consistency hypothesis and common feature representation hypothesis.  
🔥 The working mechanism of MVC-Net is demonstrated from three perspectives namely diversity, complementarity, and consensus. 

## 👉 Framework
<img src="https://github.com/QinRui-k/MVC-Net/blob/main/ARCH.png">
We develop a structurally consistent model for each view (Transverse, Coronal, Sagittal) and implement a co-training strategy. This approach enables the models to utilize multi-view consistency when processing unlabeled data, leading to uniform segmentation outputs across different perspectives. During this process, the output from the model serves as a pseudo-label for the others, enhancing their learning. Our unique multi-view weighted fusion method further refines these pseudo-labels, ensuring greater accuracy. This results in the model being more effectively supervised and, consequently, improving the performance for other views. Moreover, to capitalize on the advantages of multi-view consistency in three-dimensional data during the inference stage, we introduce a multi-view fusion inference strategy that significantly boosts segmentation accuracy.


## 👉 Evaluation
<p align="center">  
Quantitative segmentation results of the CMT-TS dataset  
</p>   
<img src="https://github.com/QinRui-k/MVC-Net/assets/139854014/905c804b-03b8-4160-979a-4d48dcfab186">

<p align="center">  
Visulization in 3D
</p>   
<img src="https://github.com/user-attachments/assets/e6247f8a-963a-4b80-8e16-699a7602578f">

<p align="center">  
Visulization in 2D
</p>   
<img src="https://github.com/user-attachments/assets/7dc403ca-50d2-4432-8ce4-796c728d0b3f">

## 👉 Train
```
cd ./MVC-Net
train.py
```

## 👉 Test
```
cd ./MVC-Net
test_fuse.py
```

## 🤝 DataSet
* We have made 100 samples publicly available on Google Drive ([link](https://drive.google.com/file/d/19YRFT-gx-L7drJiubfHelX-e1o7bkgDQ/view?usp=share_link)), while the complete dataset can be obtained by contacting the authors at 18580039560@163.com for the data link.

## 🤝 Acknowledgement
* We thank all medical workers and dataset owners for making public datasets available to the community.
