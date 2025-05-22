import os
import sqlite3
import pandas as pd

# Define folder containing Excel files
folder_name = "excel files"  # Folder where Excel files are stored
script_dir = os.path.dirname(os.path.abspath(__file__))  # Script's directory
folder_path = os.path.join(script_dir, folder_name)  # Path to Excel folder
db_path = os.path.join(script_dir, "database.db")  # SQLite database path

# Ensure the folder exists
if not os.path.exists(folder_path):
    print(f"❌ Folder '{folder_name}' not found in script directory!")
    exit()

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Function to clean data
def clean_dataframe(df):
    # 1️⃣ Remove the first 16 rows
    df = df.iloc[16:].reset_index(drop=True)
    
    # Set the first row as column headers
    df.columns = df.iloc[0]  # Assign first row as column names
    df = df[1:].reset_index(drop=True)  # Remove the first row from data

    # 2️⃣ Unmerge merged cells (forward-fill missing values)
    df.ffill(inplace=True)  # Replaces the deprecated fillna(method="ffill")

    # 3️⃣ Remove empty rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # 4️⃣ Ensure column names are strings before renaming
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # 5️⃣ Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # 6️⃣ Fill remaining missing values with empty strings
    df.fillna("", inplace=True)
    # 7️⃣ Remove specified columns
    df = df.drop([
         'root', 
        # 'ge', 
        'location', 
        'organization_id', 
        'country/_region_of_organization', 
        'mandatory_(y/n)', 
        'assigned_(y/n)', 
        'user_id', 
        'modified',
        'organization_in_hierarchy',
        'top'
    ], axis=1, errors='ignore')
    if 'organization' in df.columns:
        df.rename(columns={'organization': 'organization_name'}, inplace=True)    
    return df



# Process each Excel file
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)
        print(f"Processing {file}...")

        # Load the Excel file
        xls = pd.ExcelFile(file_path)

        # Loop through all sheets
        for sheet_name in xls.sheet_names:
            print(f"  - Cleaning and converting sheet: {sheet_name}")

            # Read sheet into DataFrame
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

            # Clean the data
            df = clean_dataframe(df)

            # Save cleaned DataFrame to SQLite table
            df.to_sql(sheet_name, conn, if_exists="append", index=False)

# Commit and close the connection
conn.commit()
conn.close()

print("✅ All Excel files have been cleaned and imported into SQLite!")
