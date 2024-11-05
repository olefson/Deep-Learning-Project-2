# Deep-Learning-Project-2
This repository contains the code and documentation for Project 2: Recurrent Neural Network (RNN), which is part of my Deep Learning coursework. The project involves designing and training recurrent neural networks using GRU and LSTM architectures to predict sequences based on provided data.

# NICU Newborn Pain Level Prediction Using GRU Model

This project aims to develop a deep learning model to predict the pain level of newborns in the NICU based on their vitals. We use a 4-layer GRU (Gated Recurrent Unit) model to classify pain levels into three categories: no pain, mild pain, and severe pain. The project includes data preprocessing, model training, and a script for model testing on new data.

## Project Structure

```
Deep-Learning-Project-2/
├── Data/                        # Directory containing the dataset subfolders
│   ├── sub01/
│   ├── sub02/
│   ├── sub03/
│   └── sub05/
├── Project2_Part1_JasonOlefson.ipynb  # Jupyter notebook with training code
├── Project2 Instructions.pdf    # PDF with project instructions
├── README.md                    # Project README file
└── saved_model/                 # Directory to save the trained model
    └── project2_part1_model.h5  # Saved GRU model file
```

## Dataset Description

Each CSV file in the `Data` directory represents vitals for newborns in the NICU ward of a hospital. The columns in each CSV file are as follows:

1. `Baby_ID`: Identifier for each baby.
2. `Heart_Rate`: Heart rate of the baby.
3. `Respiratory_Rate`: Respiratory rate of the baby.
4. `Oxygen_Saturation`: Oxygen saturation level of the baby.
5. `Pain_Level`: Pain level with values:
   - 0: No pain
   - 1: Mild pain
   - 2: Severe pain
   - #: Label not collected

## Model Architecture

The GRU model has 4 layers with dropout for regularization. The model takes in the heart rate, respiratory rate, and oxygen saturation as input features to predict the pain level.

- GRU Layer 1: 16 units, return sequences enabled
- GRU Layer 2: 16 units, return sequences enabled
- GRU Layer 3: 16 units, return sequences enabled
- GRU Layer 4: 16 units, output layer with softmax activation for classification

## Data Preprocessing

1. **Missing Labels**: Rows with missing pain level labels (`#`) are dropped.
2. **Normalization**: Features (heart rate, respiratory rate, oxygen saturation) are normalized using `StandardScaler`.
3. **Splitting**: Data is split into training, validation, and test sets with an 80-10-10 split.
4. **Shaping**: Input data is reshaped to be compatible with the GRU model.

## Training

The model is trained using the following settings:
- **Optimizer**: Adam with learning rate adjustment
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 50-60 based on experimentation
- **Regularization**: Dropout layers with 20% dropout rate

Training and validation accuracy and loss are visualized over epochs to monitor performance and avoid overfitting.

## Testing

The `test_model.py` script provides functionality to load a saved model and run predictions on new test data provided in CSV format. Note that the test data will not have labels; only the input features will be used.

### Example Usage

```python
# Load and test the model on new data
test_model(data_path="path/to/test_data.csv")
```

The `test_model` function will output the predicted pain levels for each entry in the test data.

## Usage Instructions

1. **Train the Model**: Open `Project2_Part1_JasonOlefson.ipynb` and run through the cells to preprocess data, build, and train the model.
2. **Save the Model**: Use the provided `save_model` function to save the trained model.
3. **Test the Model**: Run the `test_model.py` script, passing the path to the test CSV file as an argument.

## Requirements

- Python 3.8+
- TensorFlow
- Pandas
- Numpy
- Scikit-learn

## Project Submission Requirements

According to the project instructions, the following elements are included:
- **Training Code**: Code to preprocess data, train the model, and save it.
- **Testing Script**: `test_model.py` script to load the saved model and make predictions on new data.
- **Training and Validation Results**: Plots of training and validation accuracy and loss over epochs.

## Acknowledgments

This project was developed as part of a deep learning course. Special thanks to the course instructor for guidance and to the NICU dataset providers.

---

### Author
Jason Olefson
