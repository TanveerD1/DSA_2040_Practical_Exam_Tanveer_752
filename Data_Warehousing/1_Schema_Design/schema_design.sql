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

-- Fact Table for Sales
-- This table contains the quantitative measures of sales events
-- and foreign keys linking to the dimension tables.
CREATE TABLE IF NOT EXISTS FactSales (
    SalesID INTEGER PRIMARY KEY AUTOINCREMENT,
    InvoiceNo TEXT NOT NULL,
    Quantity INTEGER,
    UnitPrice REAL,
    TotalSales REAL,
    -- Foreign Keys
    TimeID INTEGER,
    ProductID INTEGER,
    CustomerID INTEGER,
    FOREIGN KEY (TimeID) REFERENCES DimTime(TimeID),
    FOREIGN KEY (ProductID) REFERENCES DimProduct(ProductID),
    FOREIGN KEY (CustomerID) REFERENCES DimCustomer(CustomerID)
);
