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
    t.Year,
    t.Month,
    SUM(fs.TotalSales) AS MonthlySales
FROM
    FactSales fs
JOIN
    DimTime t ON fs.TimeID = t.TimeID
JOIN
    DimCustomer c ON fs.CustomerID = c.CustomerID
WHERE
    c.Country = 'United Kingdom'
GROUP BY
    t.Year, t.Month
ORDER BY
    t.Year, t.Month;


-- Query 3: Slice
-- Objective: Isolate sales data for a specific product category ('Decor') to analyze its performance.
-- This query slices the data cube to show only the 'Decor' category.
SELECT
    p.Category,
    SUM(fs.TotalSales) AS TotalSalesForCategory
FROM
    FactSales fs
JOIN
    DimProduct p ON fs.ProductID = p.ProductID
WHERE
    p.Category = 'Decor'
GROUP BY
    p.Category;

