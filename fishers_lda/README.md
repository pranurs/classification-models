# Fisher's Linear Discriminant Analysis

## Objective

To implement Fisher's Linear Discriminant Analysis from scratch. Fisher's LDA finds the optimal direction *w* along which data, when projected, may be separated optimally. This implementation transforms the data to a single dimension, and the separator here is then the *discriminant point*, which decides the predicted class labels.

The visualization of projected points, the normal distributions of the projected classes and the discriminant point is performed for improving clarity and intuition.

---

## Dataset Description

The implementation is tested on two datasets, the first one containing 2 features for every data point (2D), and the second containing 3 features for each data example (3D).

2D dataset : `data/a1_d1.csv`

3D dataset : `data/a1_d2.csv`

Both datasets consist of 1000 data points each. Each row in each dataset corresponds to a single data point, with the last column of each row indicating the class label, which in this case, is either 0 (negative class) or 1 (positive class). The rest of the columns in each row, contain values for the features.

The algorithm is evaluated on each of these datasets independently.

**Metrics used for evaluation :**

 - Accuracy = $\frac{Number \; of \; correctly \; classified \; 
 data \; instances}{Total \; number \; of \;  data \; points}$

 - F1-Score = $\frac{2 * recall * precision}{recall + precision}$

---

## Usage

1. Edit the path of dataset on which to test the implementation in `main.py`.

2. Run

   ```shell
   $ python main.py
   ```
