# 🌟 Celebrities Face Recognition

## 📖 Project Overview
This repository provides an end-to-end solution for **Celebrity Face Recognition** using Deep Learning. It leverages the powerful **ArcFace** model (via the DeepFace framework) to generate facial embeddings and perform similarity matching. The project is split into two main components: a Jupyter Notebook for processing the dataset and building an embeddings gallery, and a sleek, fully-featured **Tkinter Desktop Application** for real-time inference without needing an active internet connection.

## ✨ Key Features
* **Deep Learning Embeddings:** Utilizes ArcFace weights to extract highly accurate facial representations.
* **31 Celebrities Supported:** Pre-configured and trained to recognize 31 prominent global figures and celebrities.
* **Offline Desktop GUI:** Features a custom-built Tkinter application that allows you to load images and predict faces entirely offline.
* **Cosine Similarity Matching:** Uses `scikit-learn` to calculate cosine similarity between the input image embedding and the pre-computed gallery to find the closest match.
* **Confidence Thresholding:** Implements a cosine similarity threshold to distinguish between a "known" celebrity and an "unknown" face.

## 🛠️ Technologies & Dependencies
To run the code, you will need **Python 3** and the following libraries:
* `deepface`
* `tf-keras`
* `opencv-python` (cv2)
* `pillow` (PIL)
* `scikit-learn`
* `numpy`

## 📁 Repository Structure
* **`Train_Code.ipynb`**: A Jupyter Notebook designed to run on a GPU environment (like Google Colab). It downloads the ArcFace model, iterates through your image dataset, extracts embeddings, tests accuracy on random images, and exports the gallery files.
* **`tkinter_interface.py`**: The main desktop application script. It provides the graphical user interface for loading images and displaying recognition results.
* **`celeb_names.json`**: A JSON list mapping the gallery indices to the string names of the 31 celebrities.
* **`gallery.pkl` & `gallery_raw.pkl`**: Pickled Python objects storing the processed embeddings mapping.
* **`gallery_matrix.npy`**: A highly optimized Numpy matrix containing the pre-calculated face embeddings for fast cosine similarity lookups during inference.

## 🎭 Supported Celebrities
The base dataset and model have been curated to recognize the following individuals:

Akshay Kumar, Alexandra Daddario, Alia Bhatt, Amitabh Bachchan, Andy Samberg, Anushka Sharma, Billie Eilish, Brad Pitt, Camila Cabello, Charlize Theron, Claire Holt, Courtney Cox, Dwayne Johnson, Elizabeth Olsen, Ellen Degeneres, Henry Cavill, Hrithik Roshan, Hugh Jackman, Jessica Alba, Kashyap, Lisa Kudrow, Margot Robbie, Marmik, Natalie Portman, Priyanka Chopra, Robert Downey Jr, Roger Federer, Tom Cruise, Vijay Deverakonda, Virat Kohli, and Zac Efron.

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amnamine/celebritiesfacerecognition.git
   cd celebritiesfacerecognition
   ```

2. **Install Required Libraries:**
   ```bash
   pip install deepface tf-keras opencv-python pillow scikit-learn numpy
   ```

3. **Ensure Required Assets are Present:**
   For the desktop app to function offline, ensure that the exported output files (`celeb_names.json`, `gallery.pkl`, `gallery_matrix.npy`) and the DeepFace `arcface_weights.h5` are placed in the same folder as `tkinter_interface.py`.

## 🚀 How to Use

### 1. Running the Desktop GUI (Inference)
Simply run the Python script to launch the interface:
```bash
python tkinter_interface.py
```
* Click **Load Image** to select an image from your local drive.
* Click **Predict** to initialize the ArcFace model, extract the features of your loaded image, and display the predicted celebrity along with the confidence/similarity score. 
* Use the **Reset** button to clear the active image and try another one.

### 2. Building a New Gallery (Training / Notebook)
If you wish to add new celebrities or rebuild the dataset, use `Train_Code.ipynb`.
* Open the notebook in Google Colab (A T4 GPU environment is recommended for speed).
* Mount your Google Drive to provide the dataset path containing folders named after your target celebrities.
* Run all cells. The script will use DeepFace to build a gallery in `~821 seconds` (depending on the dataset size), test it on random images, and export the `.npy`, `.pkl`, and `.json` files.

## 🧠 Under the Hood (How it Works)
1. **Extraction:** Using the `DeepFace` library loaded with `ArcFace` weights, a neural network analyzes an image and outputs a unique, high-dimensional numerical vector (embedding) that mathematically describes facial characteristics. 
2. **Gallery Mapping:** During the processing stage, embeddings of known dataset images are compiled into a `gallery_matrix.npy` matrix. 
3. **Similarity Search:** When a user uploads a test image via the Tkinter app, the system gets its ArcFace embedding and applies **Cosine Similarity** against the pre-saved gallery matrix. The top match corresponding to the celebrity's index is parsed through `celeb_names.json` and served back to the UI.
