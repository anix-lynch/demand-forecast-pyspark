# 📊 Demand Forecast with PySpark

**Time series forecasting model to predict product quantities sold using PySpark MLlib**

![Retail Analytics](iStock-1249219777.jpg)

---

## 🎯 Objective

Analyze the `Online Retail.csv` dataset and build a forecasting model to predict `Quantity` of products sold using:
- **Train/Test Split:** Data ≤ `2011-09-25` (train) | Data > `2011-09-25` (test)
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Forecast Target:** Week 39 of 2011 total units

---

## 📦 Dataset

**Source:** `Online Retail.csv`

**Key Features:**
- `InvoiceNo` - Transaction identifier
- `StockCode` - Product code
- `Description` - Product name
- `Quantity` - Units sold (target variable)
- `InvoiceDate` - Transaction timestamp
- `UnitPrice` - Price per unit
- `CustomerID` - Customer identifier
- `Country` - Customer location

**Size:** 384,723+ records

---

## 🚀 Workflow

### 1️⃣ Data Preparation
```python
# Load and clean data
df = spark.read.csv("Online Retail.csv", header=True, inferSchema=True)

# Parse dates and filter valid records
df = df.withColumn("InvoiceDate", to_timestamp("InvoiceDate", "M/d/yyyy H:mm"))
df = df.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))
```

### 2️⃣ Train/Test Split
```python
# Split based on date: 2011-09-25
split_date = "2011-09-25"
train = df.filter(col("InvoiceDate") <= split_date)
test = df.filter(col("InvoiceDate") > split_date)

# Aggregate daily data by Country and StockCode
pd_daily_train_data = train.groupBy("Country", "StockCode", "InvoiceDate") \
    .agg(sum("Quantity").alias("Quantity")) \
    .toPandas()
```

### 3️⃣ Feature Engineering
```python
# Extract temporal features
df = df.withColumn("Year", year("InvoiceDate"))
df = df.withColumn("Month", month("InvoiceDate"))
df = df.withColumn("DayOfWeek", dayofweek("InvoiceDate"))
df = df.withColumn("Week", weekofyear("InvoiceDate"))

# Aggregate by week for forecasting
weekly_data = df.groupBy("Year", "Week") \
    .agg(sum("Quantity").alias("TotalQuantity"))
```

### 4️⃣ Model Training
```python
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler

# Assemble features
assembler = VectorAssembler(
    inputCols=["Week", "Year", "DayOfWeek"],
    outputCol="features"
)

# Train model (e.g., Random Forest)
rf = RandomForestRegressor(featuresCol="features", labelCol="Quantity")
model = rf.fit(train_vector)
```

### 5️⃣ Evaluation
```python
# Predict on test set
predictions = model.transform(test_vector)

# Calculate MAE
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="Quantity", metricName="mae")
mae = evaluator.evaluate(predictions)
```

### 6️⃣ Week 39 Forecast
```python
# Filter predictions for week 39, 2011
week39_pred = predictions.filter((col("Week") == 39) & (col("Year") == 2011))
quantity_sold_w39 = int(week39_pred.agg(sum("prediction")).collect()[0][0])
```

---

## 📈 Key Outputs

| Variable | Description | Type |
|----------|-------------|------|
| `pd_daily_train_data` | Training set with Country, StockCode, InvoiceDate, Quantity | pandas DataFrame |
| `mae` | Mean Absolute Error on test set | float |
| `quantity_sold_w39` | Predicted units sold in week 39, 2011 | int |

---

## 🛠️ Tech Stack

- **PySpark** - Distributed data processing
- **MLlib** - Machine learning pipelines
- **Pandas** - DataFrame operations
- **Matplotlib/Seaborn** - Visualization

---

## 🎯 Results

```python
# Example outputs
mae = 245.67  # Mean Absolute Error
quantity_sold_w39 = 158420  # Week 39 forecast
```

---

## 📂 Repository Structure

```
demand-forecast-pyspark/
├── demand forecast pyspark.ipynb  # Main notebook
├── Online Retail.csv              # Dataset
├── iStock-1249219777.jpg          # Banner image
└── README.md                       # This file
```

---

## 🚦 Getting Started

```bash
# Clone repository
git clone https://github.com/anix-lynch/demand-forecast-pyspark.git
cd demand-forecast-pyspark

# Launch Jupyter
jupyter notebook "demand forecast pyspark.ipynb"
```

**Requirements:**
```bash
pip install pyspark pandas matplotlib seaborn
```

---

## 📊 Business Impact

- **Inventory Optimization:** Reduce stockouts and overstock
- **Revenue Forecasting:** Predict sales trends by product/region
- **Resource Planning:** Align staffing with demand peaks

---

## 📄 License

MIT License - Open for educational and commercial use

---

**Built with ☕ and PySpark**
