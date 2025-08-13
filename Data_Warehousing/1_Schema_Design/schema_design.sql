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
-- Populate DimTime
INSERT INTO DimTime (InvoiceDate, Day, Month, Quarter, Year) VALUES
    ('2025-01-01', 1, 1, 1, 2025),
    ('2025-01-15', 15, 1, 1, 2025),
    ('2025-02-01', 1, 2, 1, 2025),
    ('2025-03-15', 15, 3, 1, 2025),
    ('2025-04-01', 1, 4, 2, 2025),
    ('2025-04-15', 15, 4, 2, 2025),
    ('2025-05-01', 1, 5, 2, 2025),
    ('2025-06-15', 15, 6, 2, 2025),
    ('2025-07-01', 1, 7, 3, 2025),
    ('2025-07-15', 15, 7, 3, 2025),
    ('2025-08-01', 1, 8, 3, 2025),
    ('2025-09-15', 15, 9, 3, 2025),
    ('2025-10-01', 1, 10, 4, 2025),
    ('2025-11-15', 15, 11, 4, 2025),
    ('2025-12-01', 1, 12, 4, 2025);

-- Populate DimProduct
INSERT INTO DimProduct (StockCode, Description, Category) VALUES
    ('P001', 'Laptop', 'Electronics'),
    ('P002', 'Smartphone', 'Electronics'),
    ('P003', 'T-Shirt', 'Clothing'),
    ('P004', 'Jeans', 'Clothing'),
    ('P005', 'Coffee Maker', 'Appliances'),
    ('P006', 'Toaster', 'Appliances'),
    ('P007', 'Running Shoes', 'Footwear'),
    ('P008', 'Backpack', 'Accessories'),
    ('P009', 'Watch', 'Accessories'),
    ('P010', 'Headphones', 'Electronics'),
    ('P011', 'Dress', 'Clothing'),
    ('P012', 'Microwave', 'Appliances'),
    ('P013', 'Socks', 'Clothing'),
    ('P014', 'Tablet', 'Electronics'),
    ('P015', 'Wallet', 'Accessories');

-- Populate DimCustomer
INSERT INTO DimCustomer (CustomerID, Country) VALUES
    (1001, 'USA'),
    (1002, 'UK'),
    (1003, 'Canada'),
    (1004, 'Australia'),
    (1005, 'Germany'),
    (1006, 'France'),
    (1007, 'Japan'),
    (1008, 'Italy'),
    (1009, 'Spain'),
    (1010, 'Brazil'),
    (1011, 'Mexico'),
    (1012, 'India'),
    (1013, 'China'),
    (1014, 'Singapore'),
    (1015, 'UAE');

-- Populate FactSales
INSERT INTO FactSales (InvoiceNo, Quantity, UnitPrice, TotalSales, TimeID, ProductID, CustomerID) VALUES
    ('INV001', 2, 999.99, 1999.98, 1, 1, 1001),
    ('INV002', 1, 699.99, 699.99, 2, 2, 1002),
    ('INV003', 3, 29.99, 89.97, 3, 3, 1003),
    ('INV004', 2, 79.99, 159.98, 4, 4, 1004),
    ('INV005', 1, 89.99, 89.99, 5, 5, 1005),
    ('INV006', 4, 39.99, 159.96, 6, 6, 1006),
    ('INV007', 2, 129.99, 259.98, 7, 7, 1007),
    ('INV008', 1, 49.99, 49.99, 8, 8, 1008),
    ('INV009', 3, 199.99, 599.97, 9, 9, 1009),
    ('INV010', 2, 149.99, 299.98, 10, 10, 1010),
    ('INV011', 1, 89.99, 89.99, 11, 11, 1011),
    ('INV012', 2, 299.99, 599.98, 12, 12, 1012),
    ('INV013', 5, 9.99, 49.95, 13, 13, 1013),
    ('INV014', 1, 499.99, 499.99, 14, 14, 1014),
    ('INV015', 2, 39.99, 79.98, 15, 15, 1015);