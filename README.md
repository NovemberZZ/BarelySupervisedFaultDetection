# BarelySupervisedFaultDetection

This project provides the code, data, and tools involved in the manuscript "3D Seismic Fault Detection Using Fault Orthogonal Annotation and Barely Supervised Learning."

Download link for related data and tools:
https://drive.google.com/drive/folders/1u5p5Zd0M4e9yfvZalcgaMZnP313aJNaI

# PART1：Orthogonal Annotation (Orthogonal_Annotation)

The orthogonal annotation tool is developed using MATLAB-GUI. If you have MATLAB installed on your computer, you can right-click on "orthogonal_annotation_tool.m" and select "Run."

If MATLAB is not installed on your computer, we also provide an executable (exe) version for Windows users. However, you will need to install the MATLAB Runtime environment first.

![fig2](https://github.com/user-attachments/assets/a4e0283f-d4eb-4685-a8af-983eaf6e8d9c)

This tool is designed for orthogonal annotation of three-dimensional binary (.bin) seismic data. If you need to read data in other formats, you can modify lines 122 to 134 in "orthogonal_annotation_tool.m" to suit your specific requirements. In the Google Drive link, we provide the three-dimensional seismic samples clipped and preprocessed based on Poseidon, which you can directly use for orthogonal annotation.

It should be noted that this tool has not undergone extensive and rigorous stability testing, and its functionality and implementation are relatively simple. You are welcome to modify it based on your specific requirements.

# PART2：Registration (Registration_Module)

We offer the trained seismic fault registration network model files, as well as the related code and data.
Copy the orthogonal annotation images from the previous stage into the Registration_Module/reg_data_Poseidon and run "register.py" directly.

# PART3：BarelySupervisedLearning (Dense_Sparse_Co_Training)

Copy the 3D pseudo-labels obtained from the previous stage into the Dense_Sparse_Co_Training/datasets_Poseidon/labeled folder.
Then, run the main function (Main.py) to begin training.

# Note:
Please ensure that the input and output paths in the codes are correct.
Environment: Both Windows and Linux are supported, with Python 3.8, PyTorch 2.1.1, CUDA 12.1, and GPU NVIDIA A800 (or RTX 3090).
