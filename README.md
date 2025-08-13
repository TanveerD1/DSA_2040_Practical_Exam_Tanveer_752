# DSA 2040 Practical Exam - Implementation Steps
# *Tanveer 752*
## 1. Data Preprocessing

### Initial Setup
I first created the project structure and installed required libraries:

```bash
mkdir -p Data_Mining/{1_Preprocessing,2_Clustering}/preprocessed_data
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Section 1: Data Warehousing
### Task1: Data Warehousing Design
For this task I designed a data warehouse for a retail company that sells products across categories.

#### Designing a star schema
Below is a chunk of code used to create a star schema with a fact table and multiple dimension tables:

```sql
-- Dimension Table for Time
-- This table stores date and time attributes for analysis over time.
CREATE TABLE IF NOT EXISTS DimTime (
    TimeID INTEGER PRIMARY KEY AUTOINCREMENT,
    InvoiceDate TEXT NOT NULL,
    Day INTEGER,
    Month INTEGER,
    Quarter INTEGER,
    Year INTEGER
);

-- Dimension Table for Products
-- This table holds descriptive information about each product.
CREATE TABLE IF NOT EXISTS DimProduct (
    ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
    StockCode TEXT UNIQUE NOT NULL,
    Description TEXT,
    Category TEXT -- We will generate this during ETL
);

-- Dimension Table for Customers
-- This table stores information about each customer.
CREATE TABLE IF NOT EXISTS DimCustomer (
    CustomerID INTEGER PRIMARY KEY, -- Using the original CustomerID from the dataset
    Country TEXT
);
```
#### Star Diagram
Below is the star schema diagram representing the design:
![Star Schema Diagram](Data_Warehousing/1_Schema_Design/star_schema_diagram.png)

# Explanation:
- **Fact Table**: Contains foreign keys to dimension tables and measures (e.g., sales amount).
- **Dimension Tables**: Provide descriptive attributes for analysis.    
- **Star Schema**: Simplifies queries by allowing joins between the fact table and dimension tables.    
- **Speed & Efficiency for Analysis**: Fact table (FactSales) stores the core measurable events — sales quantities, prices, totals. Dimension tables (DimTime, DimProduct, DimCustomer) hold descriptive context. When you query, the database only needs to join a central fact table with small, indexed dimension tables — this is faster than joining multiple big transactional tables.
- **Simpler Queries**: With a star schema, analysts can write queries using a consistent pattern: Fact table → join → dimensions No need to navigate messy transactional relationships. This makes it easier to understand and maintain.

### Tables in the retail database
Checking using python to ensure the tables were created correctly:

```python
print("\nTables in the database:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
for (table_name,) in cursor.fetchall():
    print(f"\n=== {table_name} ===")
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5;", conn)
    display(df)

    print("\nSchema:")
    cursor.execute(f"PRAGMA table_info({table_name})")
    for column in cursor.fetchall():
        print(f"Column: {column[1]}, Type: {column[2]}")
    print("-" * 50)
```
### Filling in Tables
Using SQL to populate the tables with data from the retail dataset game me the following:
Dim Product:
![alt text](image.png)

Dim Customer:
![alt text](image-1.png)


### Task2: ETL Process Implementation
I chose to implement the ETL process using Python and the online retail dataset. The steps included:
1. **Extract**: Load the dataset from a the online source.
```python
df = pd.read_excel(DATASET_URL)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully extracted {len(df)} rows.")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    original_data_path = "online_retail_original.csv"
    df.to_csv(original_data_path, index=False)
    display(df.head())
```
which resulted in:

[2023-10-01 12:00:00] Successfully extracted 541909 rows.
Dataset shape: (541909, 8)
![alt text](image-2.png)

2. **Transform**: Clean and prepare the data, including handling missing values and generating new columns.
This included:
   - Removing rows with missing values in key columns.
   - Converting date columns to datetime format.
   - Generating new columns like `TotalPrice` and `InvoiceDate`.
   - Filtering out cancelled transactions.

```python

3. **Load**: Insert the transformed data into the SQLite database.  




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
