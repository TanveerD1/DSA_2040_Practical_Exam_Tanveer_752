# DSA 2040 Practical Exam - Implementation Steps

## 1. Data Preprocessing

### Initial Setup
I first created the project structure and installed required libraries:

```bash
mkdir -p Data_Mining/{1_Preprocessing,2_Clustering}/preprocessed_data
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Data Loading and Preprocessing
I implemented the preprocessing in `preprocessing_iris.ipynb`:

```python
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
```

### Missing Value Analysis
Results showed no missing values in the dataset:
```python
print("Missing values:\n", df.isnull().sum())
```
Output:
```
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
species              0
dtype: int64
```

### Feature Normalization
Applied Min-Max scaling to standardize features:
```python
scaler = MinMaxScaler()
df[iris.feature_names] = scaler.fit_transform(df[iris.feature_names])
```

### Visualizations

#### 1. Pairplot Analysis
![Iris Pairplot](Data_Mining/1_Preprocessing/iris_pairplot.png)

Key findings:
- Clear separation between species clusters
- Petal measurements show strongest differentiation
- Setosa species most distinctly separated

#### 2. Correlation Heatmap
![Correlation Heatmap](Data_Mining/1_Preprocessing/iris_heatmap.png)

Correlation findings:
- Strongest correlation: petal length vs petal width (0.96)
- Moderate correlation: sepal length vs petal features (~0.85)
- Weakest correlation: sepal width vs other features

#### 3. Feature Distribution
![Feature Boxplots](Data_Mining/1_Preprocessing/iris_boxplots.png)

Distribution insights:
- Setosa shows distinct measurements
- Few outliers in sepal width
- Clear size progression across species

## 2. Clustering Analysis

### K-Means Implementation
```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Evaluate clustering
ari = adjusted_rand_score(df['species'], df['cluster'])
print(f"Adjusted Rand Index (k=3): {ari:.2f}")
```
Result: ARI = 0.73 (showing strong alignment with true species)

### Elbow Method Analysis
![Elbow Plot](Data_Mining/2_Clustering/elbow_plot.png)

Findings:
- Clear elbow point at k=3
- Confirms optimal cluster number matches actual species count
- Diminishing returns after k=3

### Cluster Performance Comparison
```python
Results for different k values:
k=2: ARI = 0.57
k=3: ARI = 0.73 (optimal)
k=4: ARI = 0.65
```

### Final Cluster Visualization
![Cluster Plot](Data_Mining/2_Clustering/cluster_scatter.png)

Observations:
- High accuracy in species separation
- Some overlap between versicolor and virginica
- Perfect separation of setosa cluster

## 3. Data Storage
Final preprocessed data saved for future use:
```python
# Create directory if needed
output_dir = Path('preprocessed_data')
output_dir.mkdir(exist_ok=True)

# Save processed data
df.to_csv('preprocessed_data/iris_processed.csv', index=False)
```

## Running the Code
1. Clone repository:
```bash
git clone https://github.com/yourusername/DSA_2040_Practical_Exam_Tanveer_752.git
cd DSA_2040_Practical_Exam_Tanveer_752
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run notebooks in order:
- First: `Data_Mining/1_Preprocessing/preprocessing_iris.ipynb`
- Then: `Data_Mining/2_Clustering/clustering_iris.ipynb`

## Repository Structure
```
DSA_2040_Practical_Exam_Tanveer_752/
├── Data_Mining/
│   ├── 1_Preprocessing/
│   │   ├── preprocessing_iris.ipynb
│   │   ├── preprocessed_data/
│   │   └── iris_*.png
│   └── 2_Clustering/
│       ├── clustering_iris.ipynb
│       └── cluster_*.png
```
