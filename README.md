# **Detection and Classification of Kidney Diseases Using CT-Scanned Images**  

## **Project Description**  
This project aims to detect and classify kidney diseases (Cyst, Normal, Stone, Tumor) from CT-scanned images. Leveraging machine learning and deep learning techniques, the project automates the classification process to assist medical professionals in diagnostics.  

## **Dataset**  
- **Dataset Name**: CT-KIDNEY-DATASET  
- **Categories**:  
  1. Cyst  
  2. Normal  
  3. Stone  
  4. Tumor  
- **Structure**:  
  The dataset is organized into subdirectories for each class. Each subdirectory contains CT scan images corresponding to its label.
  
![Screenshot 2024-12-06 102055](https://github.com/user-attachments/assets/3430ff13-7e3e-4cab-9d6a-599811915a35)

## **Requirements**  
### **Python Libraries**  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `tensorflow`  
- `keras`  
- `scikit-learn`  
- `imblearn` (for SMOTE)  
- `Pillow` (for image processing)  
- `os` and `shutil` (for file operations)  

Install all dependencies with:  
```bash  
pip install -r requirements.txt  
```  

### **Folder Structure**  
```  
project_root/  
│  
├── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/  
│   ├── Cyst/  
│   ├── Normal/  
│   ├── Stone/  
│   └── Tumor/  
│  
├── notebooks/  
│   ├── data_visualization.ipynb  
│   ├── model_training.ipynb  
│   └── evaluation.ipynb  
│  
├── models/  
│   └── saved_model.h5  
│  
├── README.md  
├── requirements.txt  
└── main.py  
```  

## **Workflow**  
### **1. Data Preprocessing**  
- **Image Resizing**: All images are resized to a consistent shape (e.g., 32x32).  
- **Normalization**: Pixel values are normalized to a range of [0, 1].  
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling Technique) is used to balance the dataset.  

### **2. Data Visualization**  
- Distribution of classes.  
- Examples of CT-scanned images for each category.  

### **3. Model Training**  
- **Model Architecture**: Convolutional Neural Networks (CNN).  
- **Variants**:  
  - Base Model  
  - Dropout for regularization.  
  - Hyperparameter-tuned models.  
- **Optimizer**: Adam.  
- **Loss Function**: Categorical Cross-Entropy.  

### **4. Evaluation**  
- Accuracy and F1 Score.  
- Confusion Matrix to visualize class-wise performance.  

## **How to Run**  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/Oulakbir/Kidney_Deseases_Detection.git  
   ```  
2. Navigate to the project directory:  
   ```bash  
   cd Kidney_Deseases_Detection  
   ```  
3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
4. Run the main script for training:  
   ```bash  
   python main.py  
   ```  
5. View results and evaluation in the `evaluation.ipynb` notebook.  

## **Results**  
| Model       | Accuracy | F1 Score |  
|-------------|----------|----------|  
| Base CNN    | 85%      | 0.84     |  
| CNN + Dropout | 88%      | 0.87     |  
| Tuned CNN   | 91%      | 0.90     |  

Or Run the app.py to use the streamlit application:


![Screenshot 2024-12-06 101933](https://github.com/user-attachments/assets/361b795e-4433-4ebf-9ed7-b3e33c012685)

## **Future Work**  
- Explore advanced architectures like ResNet or EfficientNet.  
- Extend the dataset with additional cases.  
- Implement real-time prediction for CT-scanned images.  

## **Contributors**  
- **Your Name**: [LinkedIn](https://www.linkedin.com/in/ilham-oulakbir-892b50202/) | [GitHub](https://github.com/Oulakbir)  

## **License**  
This project is licensed under the MIT License. See the `LICENSE` file for more details.  
