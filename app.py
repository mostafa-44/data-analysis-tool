import sklearn as sk
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

def load_file():
    """Open file dialog and load the selected file into a DataFrame"""
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog and store selected path
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("JSON files", "*.json"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected")
        return None
        
    try:
        # Try to read different file formats
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise Exception("Unsupported file format")
            
        print(f"Successfully loaded {file_path}")
        return df
        
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

df = load_file()

if df is None:
    exit()

print("\nFirst few rows of the data:")
pd.set_option('display.max_rows', 10000)
    
print(df)

# Get list of available columns
print("\nAvailable columns in the dataset:")
for idx, column in enumerate(df.columns, 1):
    print(f"{idx}. {column}")

# Ask user to select columns for analysis
while True:
    try:
        selected_columns = input("\nEnter column numbers to analyze (separated by commas, e.g., 1,2,3): ")
        selected_indices = [int(x.strip()) - 1 for x in selected_columns.split(',')]
        selected_cols = df.columns[selected_indices].tolist()
        
        # Verify selections exist in dataframe
        if all(col in df.columns for col in selected_cols):
            print("\nYou selected these columns:")
            for col in selected_cols:
                print(f"- {col}")
            break
        else:
            print("Invalid column number(s). Please try again.")
    except (ValueError, IndexError):
        print("Invalid input. Please enter valid column numbers separated by commas.")

# Ask user to select aggregation function
print("\nAvailable aggregation functions:")
print("1. Mean")
print("2. Median")
print("3. Sum") 
print("4. Count")
print("5. Standard Deviation")
print("6. Minimum")
print("7. Maximum")

# Map user choice to aggregation function
agg_functions = {
    1: 'mean',
    2: 'median', 
    3: 'sum',
    4: 'count',
    5: 'std',
    6: 'min',
    7: 'max'
}

print("\nSelect aggregation functions (enter numbers separated by commas, e.g., 1,3,7):")
while True:
    try:
        agg_choices = input("\nEnter your choices (1-7): ").split(',')
        agg_choices = [int(choice.strip()) for choice in agg_choices]
        if all(1 <= choice <= 7 for choice in agg_choices):
            selected_aggs = [agg_functions[choice] for choice in agg_choices]
            break
        else:
            print("Please enter valid numbers between 1 and 7")
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")

# Ask user to select column for aggregation
print("\nSelect column for aggregation:")
for idx, col in enumerate(selected_cols, 1):
    print(f"{idx}. {col}")

while True:
    try:
        agg_col_idx = int(input("\nEnter column number to aggregate: "))
        if 1 <= agg_col_idx <= len(selected_cols):
            agg_column = selected_cols[agg_col_idx - 1]
            break
        else:
            print(f"Please enter a number between 1 and {len(selected_cols)}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Create a new dataframe with only selected columns
df_selected = df[selected_cols]
# Reset index and drop it to remove the index column
df_selected = df_selected.reset_index(drop=True)
print("\nAnalysis will be performed on the following data:")
df_selected = df_selected.set_index(df_selected.columns[0])
print(df_selected.head())

# Perform multiple aggregations
for agg in selected_aggs:
    result = getattr(df_selected[agg_column], agg)()
    print(f"\n{agg.capitalize()} of {agg_column}: {result:.2f}")
    
    # For min and max, show all columns for those rows
    if agg in ['min', 'max']:
        if agg == 'min':
            extreme_value = df_selected[agg_column].min()
        else:  # max
            extreme_value = df_selected[agg_column].max()
            
        extreme_rows = df[df[agg_column] == extreme_value]
        print(f"\n{agg} value of {agg_column}:")
        print(extreme_rows)

# After displaying available columns and before asking for visualization
print("\nWould you like to:")
print("1. Perform regular aggregation")
print("2. Analyze top 10 values")
print("3. Create Pivot Table")

while True:
    try:
        analysis_choice = int(input("\nEnter your choice (1-3): "))
        if 1 <= analysis_choice <= 3:
            break
        else:
            print("Please enter a number between 1 and 3")
    except ValueError:
        print("Invalid input. Please enter a number.")

if analysis_choice == 1:
    # Ask user to select chart type
    print("\nAvailable chart types:")
    print("1. Line Chart")
    print("2. Bar Chart")
    print("3. Scatter Plot")
    print("4. Histogram")
    print("5. Box Plot")
    print("6. Pie Chart")

    while True:
        try:
            chart_type = int(input("\nSelect chart type (enter number 1-6): "))
            if 1 <= chart_type <= 6:
                break
            else:
                print("Please enter a number between 1 and 6")
        except ValueError:
            print("Invalid input. Please enter a number.")
        

    # Create the selected chart type
    plt.style.use('default')  # Use default style instead of seaborn
    sns.set_style("whitegrid")  # Set seaborn style this way
    plt.figure(figsize=(12, 7))  # Larger figure size
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_cols)))  # Generate beautiful color palette

    if chart_type == 1:  # Line Chart
        if len(selected_cols) >= 2:
            x_col = selected_cols[0]
            y_col = selected_cols[1]
            plt.plot(df_selected[x_col], df_selected[y_col], linewidth=2.5, 
                    color=colors[1], marker='o', markersize=6, 
                    label=f'{y_col} vs {x_col}')
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
        else:
            for idx, col in enumerate(selected_cols):
                plt.plot(df_selected[col], label=col, linewidth=2.5, 
                        color=colors[idx], marker='o', markersize=6)
        plt.title('Line Chart of Selected Columns', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

    elif chart_type == 2:  # Bar Chart
        df_selected.plot(kind='bar', color=colors, width=0.8)
        plt.title('Bar Chart of Selected Columns', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

    elif chart_type == 3:  # Scatter Plot
        if len(selected_cols) >= 2:
            plt.scatter(df_selected[selected_cols[0]], df_selected[selected_cols[1]], 
                    c=colors[0], alpha=0.6, s=100)
            plt.xlabel(selected_cols[0], fontsize=12)
            plt.ylabel(selected_cols[1], fontsize=12)
            plt.title(f'Scatter Plot: {selected_cols[0]} vs {selected_cols[1]}', 
                    fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
        else:
            print("Scatter plot requires at least 2 columns. Please select more columns.")

    elif chart_type == 4:  # Histogram
        for idx, col in enumerate(selected_cols):
            plt.hist(df_selected[col], alpha=0.7, label=col, color=colors[idx], 
                    bins=20, edgecolor='white')
        plt.title('Histogram of Selected Columns', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

    elif chart_type == 5:  # Box Plot
        df_selected.boxplot(patch_artist=True, 
                        boxprops=dict(facecolor=colors[0], alpha=0.7),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor=colors[0]))
        plt.title('Box Plot of Selected Columns', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

    else:  # Pie Chart (chart_type == 6)
        if len(selected_cols) == 1:
            values = df_selected.iloc[:, 0]
            plt.pie(values, labels=df_selected.index, autopct='%1.1f%%',
                colors=plt.cm.Set3(np.linspace(0, 1, len(values))),
                wedgeprops=dict(width=0.7, edgecolor='white'),
                textprops={'fontsize': 10})
            plt.title(f'Pie Chart of {selected_cols[0]}', fontsize=14, pad=20)
        else:
            print("Pie chart works best with a single column. Using first selected column.")
            values = df_selected.iloc[:, 0]
            plt.pie(values, labels=df_selected.index, autopct='%1.1f%%',
                colors=plt.cm.Set3(np.linspace(0, 1, len(values))),
                wedgeprops=dict(width=0.7, edgecolor='white'),
                textprops={'fontsize': 10})
            plt.title(f'Pie Chart of {selected_cols[0]}', fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()
elif analysis_choice == 2:
    # Ask user to select column for analysis
    print("\nSelect column for analysis:")
    for idx, col in enumerate(selected_cols, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            top_col_idx = int(input("\nEnter column number to analyze: "))
            if 1 <= top_col_idx <= len(selected_cols):
                top_column = selected_cols[top_col_idx - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(selected_cols)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Ask user to choose between top 5 or top 10
    print("\nWould you like to analyze:")
    print("1. Top 5 values")
    print("2. Top 10 values")

    while True:
        try:
            top_n_choice = int(input("\nEnter your choice (1-2): "))
            if 1 <= top_n_choice <= 2:
                n_values = 5 if top_n_choice == 1 else 10
                break
            else:
                print("Please enter either 1 or 2")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Calculate and display top values
    top_values = df[top_column].nlargest(n_values)
    print(f"\nTop {n_values} values in {top_column}")
    print(top_values)

    # Show all columns for the top values
    top_df = df.loc[top_values.index]
    print(f"\nAll columns for the top {n_values} values:")
    print(top_df)

    # Ask user what they want to do with the top values
    print("\nWhat would you like to do with these top values?")
    print("1. Create visualization")
    print("2. Create pivot table")
    
    while True:
        try:
            top_analysis_choice = int(input("\nEnter your choice (1-2): "))
            if 1 <= top_analysis_choice <= 2:
                break
            else:
                print("Please enter either 1 or 2")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if top_analysis_choice == 1:
        # Ask user to select chart type
        print("\nAvailable chart types:")
        print("1. Bar Chart")
        print("2. Line Chart")
        print("3. Pie Chart")
        print("4. Horizontal Bar Chart")
        print("5. Area Chart")
        print("6. Scatter Plot")

        while True:
            try:
                chart_choice = int(input("\nSelect chart type (enter number 1-6): "))
                if 1 <= chart_choice <= 6:
                    break
                else:
                    print("Please enter a number between 1 and 6")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Ask user to select columns for x and y axes
        print("\nSelect column for X-axis:")
        for idx, col in enumerate(df.columns, 1):
            print(f"{idx}. {col}")

        while True:
            try:
                x_col_idx = int(input("\nEnter column number for X-axis: "))
                if 1 <= x_col_idx <= len(df.columns):
                    x_column = df.columns[x_col_idx - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(df.columns)}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        print("\nSelect column for Y-axis:")
        for idx, col in enumerate(df.columns, 1):
            print(f"{idx}. {col}")

        while True:
            try:
                y_col_idx = int(input("\nEnter column number for Y-axis: "))
                if 1 <= y_col_idx <= len(df.columns):
                    y_column = df.columns[y_col_idx - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(df.columns)}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, n_values))

        if chart_choice == 1:  # Bar Chart
            bars = plt.bar(top_df[x_column], top_df[y_column], color=colors)
            plt.title(f'{y_column} vs {x_column} (Top {n_values})', fontsize=14, pad=20)
            plt.xlabel(x_column, fontsize=12)
            plt.ylabel(y_column, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

        elif chart_choice == 2:  # Line Chart
            plt.plot(top_df[x_column], top_df[y_column], 
                    marker='o', linewidth=2, markersize=8, color=colors[0])
            plt.title(f'{y_column} vs {x_column} (Top {n_values})', fontsize=14, pad=20)
            plt.xlabel(x_column, fontsize=12)
            plt.ylabel(y_column, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for i, (x, y) in enumerate(zip(top_df[x_column], top_df[y_column])):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

        elif chart_choice == 3:  # Pie Chart
            plt.pie(top_df[y_column], labels=top_df[x_column],
                autopct='%1.1f%%', colors=colors)
            plt.title(f'Distribution of {y_column} by {x_column} (Top {n_values})', 
                    fontsize=14, pad=20)

        elif chart_choice == 4:  # Horizontal Bar Chart
            bars = plt.barh(top_df[x_column], top_df[y_column], color=colors)
            plt.title(f'{y_column} vs {x_column} (Top {n_values})', fontsize=14, pad=20)
            plt.xlabel(y_column, fontsize=12)
            plt.ylabel(x_column, fontsize=12)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.2f}',
                        ha='left', va='center')

        elif chart_choice == 5:  # Area Chart
            plt.fill_between(range(len(top_df)), top_df[y_column], 
                           color=colors[0], alpha=0.3)
            plt.plot(range(len(top_df)), top_df[y_column], 
                    color=colors[0], linewidth=2)
            plt.title(f'{y_column} vs {x_column} (Top {n_values})', fontsize=14, pad=20)
            plt.xlabel(x_column, fontsize=12)
            plt.ylabel(y_column, fontsize=12)
            plt.xticks(range(len(top_df)), top_df[x_column], rotation=45, ha='right')
            
            # Add value labels
            for i, y in enumerate(top_df[y_column]):
                plt.text(i, y, f'{y:.2f}', ha='center', va='bottom')

        else:  # Scatter Plot (chart_choice == 6)
            plt.scatter(top_df[x_column], top_df[y_column], 
                    s=100, color=colors)
            plt.title(f'{y_column} vs {x_column} (Top {n_values})', fontsize=14, pad=20)
            plt.xlabel(x_column, fontsize=12)
            plt.ylabel(y_column, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for x, y in zip(top_df[x_column], top_df[y_column]):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:  # top_analysis_choice == 2
        # Create a new dataframe with only the top values
        top_df = df.loc[top_values.index]
        
        print("\nSelect column for row index (categorical data):")
        for idx, col in enumerate(selected_cols, 1):
            print(f"{idx}. {col}")

        while True:
            try:
                row_idx = int(input("\nEnter column number for row index: "))
                if 1 <= row_idx <= len(selected_cols):
                    row_column = selected_cols[row_idx - 1]
                    # Convert to string type to ensure it works as a categorical index
                    top_df[row_column] = top_df[row_column].astype(str)
                    break
                else:
                    print(f"Please enter a number between 1 and {len(selected_cols)}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        print("\nSelect column for column index (categorical data):")
        for idx, col in enumerate(selected_cols, 1):
            print(f"{idx}. {col}")

        while True:
            try:
                col_idx = int(input("\nEnter column number for column index: "))
                if 1 <= col_idx <= len(selected_cols):
                    col_column = selected_cols[col_idx - 1]
                    # Convert to string type to ensure it works as a categorical index
                    top_df[col_column] = top_df[col_column].astype(str)
                    break
                else:
                    print(f"Please enter a number between 1 and {len(selected_cols)}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        print("\nSelect column for values (numerical data):")
        for idx, col in enumerate(selected_cols, 1):
            print(f"{idx}. {col}")

        while True:
            try:
                val_idx = int(input("\nEnter column number for values: "))
                if 1 <= val_idx <= len(selected_cols):
                    val_column = selected_cols[val_idx - 1]
                    # Ensure the values column is numeric
                    top_df[val_column] = pd.to_numeric(top_df[val_column], errors='coerce')
                    break
                else:
                    print(f"Please enter a number between 1 and {len(selected_cols)}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        print("\nSelect aggregation function for pivot table:")
        print("1. Mean")
        print("2. Sum")
        print("3. Count")
        print("4. Median")
        print("5. Min")
        print("6. Max")

        while True:
            try:
                pivot_agg = int(input("\nEnter your choice (1-6): "))
                if 1 <= pivot_agg <= 6:
                    agg_func = {
                        1: 'mean',
                        2: 'sum',
                        3: 'count',
                        4: 'median',
                        5: 'min',
                        6: 'max'
                    }[pivot_agg]
                    break
                else:
                    print("Please enter a number between 1 and 6")
            except ValueError:
                print("Invalid input. Please enter a number.")

        try:
            # Create pivot table with error handling
            pivot_table = pd.pivot_table(
                top_df,
                values=val_column,
                index=row_column,
                columns=col_column,
                aggfunc=agg_func,
                fill_value=0
            )

            # Display pivot table
            print("\nPivot Table (from top values):")
            print(pivot_table)

            # Create heatmap visualization of pivot table
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', 
                    cbar_kws={'label': agg_func.capitalize()})
            plt.title(f'Pivot Table Heatmap: {agg_func.capitalize()} of {val_column} (Top Values)')
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"\nError creating pivot table: {str(e)}")
            print("Tips:")
            print("- Make sure row and column indices are categorical (text/discrete values)")
            print("- Make sure the values column contains numerical data")
            print("- Check for missing or invalid data in the selected columns")
else:  # analysis_choice == 3
    # First, let's clean and prepare the data
    df_pivot = df.copy()
    
    # Convert any potential problematic columns to string type for categorical data
    print("\nSelect column for row index (categorical data):")
    for idx, col in enumerate(selected_cols, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            row_idx = int(input("\nEnter column number for row index: "))
            if 1 <= row_idx <= len(selected_cols):
                row_column = selected_cols[row_idx - 1]
                # Convert to string type to ensure it works as a categorical index
                df_pivot[row_column] = df_pivot[row_column].astype(str)
                break
            else:
                print(f"Please enter a number between 1 and {len(selected_cols)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print("\nSelect column for column index (categorical data):")
    for idx, col in enumerate(selected_cols, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            col_idx = int(input("\nEnter column number for column index: "))
            if 1 <= col_idx <= len(selected_cols):
                col_column = selected_cols[col_idx - 1]
                # Convert to string type to ensure it works as a categorical index
                df_pivot[col_column] = df_pivot[col_column].astype(str)
                break
            else:
                print(f"Please enter a number between 1 and {len(selected_cols)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print("\nSelect column for values (numerical data):")
    for idx, col in enumerate(selected_cols, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            val_idx = int(input("\nEnter column number for values: "))
            if 1 <= val_idx <= len(selected_cols):
                val_column = selected_cols[val_idx - 1]
                # Ensure the values column is numeric
                df_pivot[val_column] = pd.to_numeric(df_pivot[val_column], errors='coerce')
                break
            else:
                print(f"Please enter a number between 1 and {len(selected_cols)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Rest of the pivot table code remains the same
    print("\nSelect aggregation function for pivot table:")
    print("1. Mean")
    print("2. Sum")
    print("3. Count")
    print("4. Median")
    print("5. Min")
    print("6. Max")

    while True:
        try:
            pivot_agg = int(input("\nEnter your choice (1-6): "))
            if 1 <= pivot_agg <= 6:
                agg_func = {
                    1: 'mean',
                    2: 'sum',
                    3: 'count',
                    4: 'median',
                    5: 'min',
                    6: 'max'
                }[pivot_agg]
                break
            else:
                print("Please enter a number between 1 and 6")
        except ValueError:
            print("Invalid input. Please enter a number.")

    try:
        # Create pivot table with error handling
        pivot_table = pd.pivot_table(
            df_pivot,
            values=val_column,
            index=row_column,
            columns=col_column,
            aggfunc=agg_func,
            fill_value=0
        )

        # Display pivot table
        print("\nPivot Table:")
        print(pivot_table)

        # Create heatmap visualization of pivot table
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': agg_func.capitalize()})
        plt.title(f'Pivot Table Heatmap: {agg_func.capitalize()} of {val_column}')
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"\nError creating pivot table: {str(e)}")
        print("Tips:")
        print("- Make sure row and column indices are categorical (text/discrete values)")
        print("- Make sure the values column contains numerical data")
        print("- Check for missing or invalid data in the selected columns")





























