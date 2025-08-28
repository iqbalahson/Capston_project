### Project: Real-time Pothole Segmentation

## Overview
This project uses the YOLOv8l model for real-time pothole detection. By identifying the precise and shape of potholes from images/videos feeds, it provides valuable data for road maintenance crews. This information helps improve public safety and assists in managing urban infrastructure. The detailed pothole data can also be applied to smart city initiatives and the development of autonomous vehicle navigation.


##  Objectives
Key milestones in this project include:
*   **Model Selection:** Chose YOLOv8l for its balance of speed and accuracy, making it suitable for real-time analysis.
*   **Dataset Curation:** Prepared and augmented a specialized dataset of pothole images for effective model training.
*   **Model Fine-Tuning:** Used transfer learning to adapt the YOLOv8l model for precise pothole segmentation.
*   **Performance Evaluation:** Assessed the model using a variety of metrics to confirm its accuracy and reliability.
*   **Inference and Validation:** Tested the trained model on new images and videos to verify its real-world performance.
*   **Application Development:** Built a real-time application to detect and measure road damage from a images/videos feed.


##  Dataset Description

###  Overview
The [**Pothole Detection for Road Safety Dataset**](https://github.com/iqbalahson/Pothole_test) is purpose-built for training YOLOv8l models to identify potholes.

###  Specifications 
-  **Class**: '- D00, D10, D20, D40, D50, D60, D70, D80' 
-  **Total Images used to trained**: 975
-  **Image Dimensions**: 640x640 pixels   
-  **Format**: YOLOv8 annotation format

###  Pre-processing
Includes auto-orientation and resizing to 640x640 for consistency.

###  Dataset Split
- **Training Set**: 233 images with augmentations.
- **Validation Set**: 342 images.

###  Augmentation on Training Set
Comprising flips, cropping, rotation, shearing, brightness, and exposure adjustments.



##  File Descriptions

- **`model/`**: Includes the best-performing fine-tuned YOLOv8 model in `.pt` (PyTorch format) used for pothole segmentation.
- **`model_training.ipynb`**: The Jupyter notebook that documents the model development pipeline, from data preparation to model evaluation and inference.
- **`detect_server_data.py`**: The Python script for deploying the YOLOv8 segmentation model to estimate road damage in real-time.
- **`data/images`**: seriese of image file used to demonstrate the application's capabilities.
- **`LICENSE`**: Outlines the terms of use for this project's resources.
- **`README.md`**: The document you are reading that provides an overview and essential details of the project.


## Instructions for file clonning and Locally Execution

To experience the full capabilities of the YOLOv8 Traffic Density Estimation project on your local machine, follow these steps:

### . Initial Setup
1. **Clone the Repository**: Start by cloning the project repository to your local system using the command below:
    ```bash
    git clone https://github.com/iqbalahson/Pothole_test.git
    ```
    
2. **Navigate to the Project Directory**: After cloning, change into the project directory with:
    ```bash
    cd Capston_Project
    ```
3. **Navigate to the Project Directory**: After cloning, change into the project directory with:
    ```bash
    python3 -m venv venv           # if not work use inbult interpretor to built create virtual environment
    source venv/bin/activate     # On Linux/Mac
    venv\Scripts\activate        # On Windows

    ```
4. **To install packages from requirements.txt on another machine:
    ```bash
    pip install -r requirements.txt
    ```

### . Exploring the Model Development Pipeline
Get hands-on with the model development process and see the results of traffic density estimation:
1. **Download the Dataset**: Access the dataset from [Google_drive](https://drive.google.com/drive/folders/1A1BI1q6ZpHQhgN9cRJl4xW3stZGxR7lR). Download and extract it to a known directory on your machine.
2. **Open the Notebook**: Launch Jupyter Notebook or JupyterLab and open `model_training.ipynb` to explore the model development pipeline.
3. **Install Dependencies**: Ensure all necessary Python libraries are installed for flawless execution.
4. **Update Paths**: Update the paths in the notebook for the dataset, sample image, and sample video to their respective locations on your local system.
5. **Run the Notebook**: Execute all cells in the notebook to step through the data preprocessing, model training, and evaluation phases.

### . steps to train data set on google colab
**Data set drive link**: download data from here:
    ```bash
   https://drive.google.com/drive/folders/1A1BI1q6ZpHQhgN9cRJl4xW3stZGxR7lR
    ```
1. **Copy data set in to drive**: Ensure it has the right path :
    ```bash
    dataset_path = '/content/drive/MyDrive/Colab_Notebooks/india_txt'
    ```
2. **Check that directory has data oploaded or not**: This will help to estimate the file and surity of data inside that folder.
    ```bash
    !cd /content/drive/MyDrive/Colab_Notebooks/india_txt/train && du -d 1 --human-readable
    ```
3. **check runtime setting**: Ensure you have opted for GPU processor:
    ```bash
    Colab>>>click on the top right corner>>>select GPU.
    ```
4. **Install Ultralytics YOLO**: Ensure you have the `ultralytics` package installed by running:
    ```bash
    !pip install ultralytics
    ```
5. **run command in jupyter file one by one**: Ensure you have followed correct steps before running:
    ```bash
   for any inquery contact iqbalahson29@gmail.com
    ```
