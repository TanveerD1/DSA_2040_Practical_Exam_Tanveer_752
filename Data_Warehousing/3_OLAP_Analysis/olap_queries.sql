-- OLAP Style Queries for the Retail Data Warehouse

-- Query 1: Roll-up
-- Objective: Aggregate total sales by country and then by quarter to see high-level performance.
-- This query rolls up sales from individual transactions to the country-quarter level.
SELECT
    c.Country,
    t.Year,
    t.Quarter,
    SUM(fs.TotalSales) AS TotalSalesAmount
FROM
    FactSales fs
JOIN
    DimCustomer c ON fs.CustomerID = c.CustomerID
JOIN
    DimTime t ON fs.TimeID = t.TimeID
GROUP BY
    c.Country, t.Year, t.Quarter
ORDER BY
    c.Country, t.Year, t.Quarter;


-- Query 2: Drill-down
-- Objective: Investigate the monthly sales performance within a specific country (e.g., 'United Kingdom').
-- This query drills down from the country level to monthly details.
SELECT 
    t.Month,
    t.Year,
    p.Category,
    SUM(f.Quantity) as TotalQuantity,
    SUM(f.TotalSales) as TotalSales
FROM FactSales f
JOIN DimCustomer c ON f.CustomerID = c.CustomerID
JOIN DimTime t ON f.TimeID = t.TimeID
JOIN DimProduct p ON f.ProductID = p.ProductID
WHERE c.Country = 'United Kingdom'
GROUP BY t.Year, t.Month, p.Category
ORDER BY t.Year, t.Month, p.Category;



-- Query 3: Slice
-- Objective: Isolate sales data for a specific product category ('Decor') to analyze its performance.
-- This query slices the data cube to show only the 'Decor' category.
SELECT 
    c.Country,
    SUM(f.TotalSales) as TotalSales
FROM FactSales f
JOIN DimCustomer c ON f.CustomerID = c.CustomerID
JOIN DimProduct p ON f.ProductID = p.ProductID
WHERE p.Category = 'Decor'
GROUP BY c.Country
ORDER BY TotalSales DESC;

