Module 1 - Data Pipeline
Setup and Execution
Install the required dependencies using pip install -r requirements.txt.
Open pipeline.ipynb and run all the cells in order.
Once the pipeline finishes, the zepto_catalogue.db database will be created automatically in the project directory.
Design Decisions
Handling Missing Data: Any rows where the price or rating could not be converted into a valid numeric value were removed. This helps keep incorrect or incomplete data from affecting the analysis.
Currency Conversion: All prices were converted using the fixed exchange rate specified for the project: 1 GBP = 105.50 INR.
Database Design: The data is divided into two tables, categories and books. They are connected using a category_id Primary Key and Foreign Key relationship. This avoids repeating category names for every book and keeps the database structure organized.