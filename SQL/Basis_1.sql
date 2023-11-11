-- Завдання 2:
-- Таблиця вин (Wine)
CREATE TABLE Wine (
    WineID INT PRIMARY KEY,
    Name NVARCHAR(255) NOT NULL,
    Description NVARCHAR(MAX),
    Type NVARCHAR(50),
    Country NVARCHAR(50),
    Price DECIMAL(10,2) 
);
-- Таблиця клієнтів (Customer)
CREATE TABLE Customer (
    CustomerID INT PRIMARY KEY,
    FirstName NVARCHAR(50),
    LastName NVARCHAR(50),
    Email NVARCHAR(100),
    Phone NVARCHAR(15),
    Address NVARCHAR(MAX)
);
-- Таблиця продаж (Sales)
CREATE TABLE Sales (
    SaleID INT PRIMARY KEY,
    WineID INT,
    SaleDate DATE,
    QuantitySold INT,
    TotalAmount DECIMAL(10,2),
    CustomerID INT,
    FOREIGN KEY (WineID) REFERENCES Wine(WineID),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);
-- Таблиця закупок (Purchases)
CREATE TABLE Purchases (
    PurchaseID INT PRIMARY KEY,
    WineID INT,
    PurchhaseDate DATE,
    QuantityPurchased INT,
    Cost DECIMAL(10,2),
    FOREIGN KEY (WineID) REFERENCES Wine(WineID)
);

-- Завдання 3:
-- Вставка в таблицю Wine
INSERT INTO Wine (WineID, Name, Description, Type, Country, Price) VALUES
    (1, 'Chateau Margaux', N'Червоне вино класу Grand Cru', N'Червоне', N'Франція', 500.00),
    (2, 'Chardonnay Reserve', N'Біле вино з дубовим смаком', N'Біле', N'США', 45.00),
    (3, 'Prosecco Extra Dry', N'Італійське ігристе вино', N'Ігристе',  N'Італія', 25.00);

-- Вставка в таблицю Customer
INSERT INTO Customer (CustomerID, FirstName, LastName, Email, Phone, Address) VALUES
    (1, N'Гаррі', N'Поттер', 'harry@example.com', '+380123456789', N'4 Привідна вулиця, Лондон'),
    (2, N'Герміона', N'Грейнджер', 'hermione@example.com', '+380987654321', N'7 Дубова вулиця, Лондон'),
    (3, N'Рон', N'Уізлі', 'ron@example.com', '+380567890123', N'12 Бурхлива вулиця, Лондон'),
    (4, N'Луна', N'Лавґуд', 'luna@example.com', '+3801111222333', N'23 Дивовижна вулиця, Лондон'),
    (5, N'Сіріус', N'Блек', 'sirius@example.com', '+3804444555566', N'10 Відьминська вулиця, Лондон');

-- Вставка в таблицю Sales
INSERT INTO Sales (SaleID, WineID, SaleDate, QuantitySold, TotalAmount, CustomerID) VALUES
    (1, 1, '2023-09-15', 3, 1500.00, 1),
    (2, 2, '2023-09-16', 5, 225.00, 2),
    (3, 3, '2023-09-17', 2, 50.00, 3);
 
 -- Вставка в таблицю Purchases
INSERT INTO Purchases (PurchaseID, WineID, PurchhaseDate, QuantityPurchased, Cost) VALUES
    (1, 1, '2023-09-10', 10, 3500.00),
    (2, 2, '2023-09-11', 20, 800.00),
    (3, 3, '2023-09-12', 15, 375.00);

-- Завдання 4:
-- Оновлення інформації
UPDATE Customer
SET Phone = '+380999888777'
WHERE FirstName = N'Сіріус' AND LastName = N'Блек';

-- Завдання 5:
-- Видалення запису
DELETE FROM Sales WHERE SaleID = 3;