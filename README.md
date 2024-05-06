# MVC-Net

We develop a structurally consistent model for each view (Transverse, Coronal, Sagittal) and implement a co-training strategy. This approach enables the models to utilize multi-view consistency when processing unlabeled data, leading to uniform segmentation outputs across different perspectives. During this process, the output from the model serves as a pseudo-label for the others, enhancing their learning. Our unique multi-view weighted fusion method further refines these pseudo-labels, ensuring greater accuracy. This results in the model being more effectively supervised and, consequently, improving the performance for other views. Moreover, to capitalize on the advantages of multi-view consistency in three-dimensional data during the inference stage, we introduce a multi-view fusion inference strategy that significantly boosts segmentation accuracy.
<img src="https://github.com/QinRui-k/MVC-Net/blob/main/ARCH.png">


# Evaluation
<img src="https://github.com/QinRui-k/MVC-Net/files/15224369/JAW-3D.pdf">
