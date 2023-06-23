import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import seaborn as sns

from search_log_manager import SearchLogManager


def read_csv_files(directory, keyword_filepath):
    """
    Read all CSV files from the specified directory and their corresponding keyword strings.

    Args:
    directory (str): The directory to read the CSV files from.
    keyword_filepath (str): The path to the CSV file containing the keyword strings.

    Returns:
    dict: A dictionary where the keys are the filenames (representing the different queries) and the values are tuples containing the pandas DataFrame and the corresponding keyword string.
    """
    # Read the keyword file into a pandas DataFrame
    keyword_df = pd.read_csv(keyword_filepath)

    # Convert the DataFrame to a dictionary
    keyword_dict = keyword_df.set_index('query_name')['Keyword String'].to_dict()

    data = {}

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(os.path.join(directory, filename))

            # Convert 'coverDate' to DateTime object
            df['coverDate'] = pd.to_datetime(df['coverDate'], format='%Y-%m-%d')

            # Extract year and create a new column 'Year'
            df['Year'] = df['coverDate'].dt.year

            # Get the name of the search query from the filename (without the ".csv" extension)
            query = os.path.splitext(filename)[0]

            # Get the corresponding keyword string from the dictionary
            keyword_string = keyword_dict.get(query, 'Unknown query')

            # Store the DataFrame and the keyword string in the dictionary
            data[query] = (df, keyword_string)

    return data


def compare_total_studies(data):
    """
    Compare the total number of studies for different search queries.

    Args:
    data (dict): A dictionary where the keys are the filenames (representing the different queries) and the values are
    tuples containing the pandas DataFrame and the corresponding keyword string.
    """

    # Initialize a list to store the results
    results = []

   # Loop through each dataframe in the dictionary
    for query, (df, keyword_string) in data.items():
        # Get the total number of studies
        total_studies = df.shape[0]

        # Add the result to the list
        results.append((query, total_studies, keyword_string))

    # Convert the data to a DataFrame
    data_df = pd.DataFrame(results, columns=['Query', 'Total Studies', 'Keyword String'])

    # Combine the query and keyword string for the legend
    data_df['Legend'] = data_df['Query'] + ' (' + data_df['Keyword String'] + ')'

    # Set the seaborn style to "whitegrid" with no vertical gridlines
    sns.set_style("whitegrid")

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 7))  # make the plot wider
    colors = cm.Blues(np.linspace(0.5, 1, len(data_df)))  # generate colors from a blue colormap

    # Plot bars with colors and a specified width
    bars = ax.bar(data_df['Query'], data_df['Total Studies'], color=colors, width=0.5)

    # Add the number of studies as text on the bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                data_df['Total Studies'][i], ha='center', va='bottom', color='black')

    plt.ylabel('Total Studies')
    plt.title('Total Studies by Query')
    plt.xticks(rotation=0)  # rotate x-axis labels to horizontal
    plt.subplots_adjust(bottom=0.3, top=0.9)  # adjust the subplot size to accommodate the legend

    # Turn off vertical grid lines
    ax.xaxis.grid(False)

    # Create a legend showing query names and keyword strings
    handles = [plt.Rectangle((0,0),1,1, color = bar.get_facecolor()) for bar in bars]
    ax.legend(handles, data_df['Legend'].tolist(), title='Query', bbox_to_anchor=(0.5, -0.15), loc='upper center',
              ncol=1, fontsize=6)

    # Save the plot as an image
    image_name = "total_studies_comparison.png"

    plt.show()


def compare_cumulative_documents_by_year(data):
    """
    Compare the cumulative number of documents by year for different search queries.

    Args:
    data (dict): A dictionary where the keys are the filenames (representing the different queries) and the values are
    tuples containing the pandas DataFrame and the corresponding keyword string.
    """

    # Set the seaborn style to "whitegrid" with no vertical gridlines
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))  # make the plot wider
    colors = cm.Blues(np.linspace(0.5, 1, len(data)))  # generate colors from a blue colormap

    max_cumulative_docs = 0  # To track the maximum cumulative document count across all data
    min_year = np.inf  # To track the minimum year across all data
    max_year = -np.inf  # To track the maximum year across all data

    # Reverse the order of the data dictionary
    data_reversed = dict(reversed(list(data.items())))

    # Loop through each dataframe in the dictionary
    for i, (query, (df, keyword_string)) in enumerate(data_reversed.items()):
        # Count the number of documents per year
        doc_counts = df['Year'].value_counts().sort_index()

        # Create a list of all years from the earliest in your data to the latest
        all_years = np.arange(df['Year'].min(), df['Year'].max() + 1)

        # Update minimum and maximum year if necessary
        if df['Year'].min() < min_year:
            min_year = df['Year'].min()
        if df['Year'].max() > max_year:
            max_year = df['Year'].max()

        # Reindex your data to include all years, filling missing years with 0
        doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

        # Calculate the cumulative sum of the documents
        cumulative_doc_counts = doc_counts_reindexed.cumsum()

        # Update maximum cumulative document count if necessary
        if cumulative_doc_counts.max() > max_cumulative_docs:
            max_cumulative_docs = cumulative_doc_counts.max()

        # Plot the data
        plt.fill_between(cumulative_doc_counts.index, cumulative_doc_counts.values,
                         color=colors[i], alpha=0.5, label=f'{query} ({keyword_string})')
        plt.plot(cumulative_doc_counts.index, cumulative_doc_counts.values,
                 color=colors[i], alpha=1)

    plt.title('Cumulative Documents by Year')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Number of Documents')

    # Set the limits of the y-axis
    plt.ylim(0, max_cumulative_docs + 5)

    # Set the limits of the x-axis
    plt.xlim(min_year, max_year)

    # Turn off vertical grid lines
    plt.gca().xaxis.grid(False)

    # Create a legend showing query names and keyword strings
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=1, fontsize=6)

    # Adjust the plot layout to make sure nothing is cropped
    plt.subplots_adjust(bottom=0.3)

    plt.show()


def compare_top_sources(data, N):
    """
    Compares the top N sources in terms of document count from multiple CSV files.

    Args:
    data (dict of pandas.DataFrame): A dictionary mapping CSV file paths to DataFrames of their data.
    N (int): The number of sources to display.

    Returns:
    None
    """

    # Empty DataFrame to hold the document counts for each source
    doc_counts_df = pd.DataFrame()

    all_source_counts = pd.Series(dtype=int)  # A series to hold document counts for all sources across CSVs

    for csv_path, df_keyword in data.items():
        df, _ = df_keyword  # unpack the tuple to separate the DataFrame and the keyword string

        # Create a dictionary with source names as keys and their counts as values
        source_counts = df['publicationName'].value_counts()

        all_source_counts = all_source_counts.add(source_counts, fill_value=0)  # Sum the counts of documents for each source

        # Add to the DataFrame
        doc_counts_df = doc_counts_df.join(source_counts.rename(csv_path), how='outer')

    # Select the top N sources
    top_sources = all_source_counts.nlargest(N).index

    # Filter the DataFrame to only include the top sources
    doc_counts_df = doc_counts_df.loc[top_sources]

    # Replace NaN values with zeros
    doc_counts_df = doc_counts_df.fillna(0)

    # Plot the data
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing
    plt.figure(figsize=(12, 6))  # Set the figure size

    # Get the 'Blues' colormap
    original_blues = plt.cm.get_cmap('Blues')

    # Create a new colormap that only uses the colors from 0.5 to 1.0 of the original colormap
    dark_blues = ListedColormap(original_blues(np.linspace(0.5, 1.0, 256)))

    doc_counts_df.plot(kind='barh', stacked=False, ax=plt.gca(), colormap=dark_blues)

    plt.title('Documents by Source')
    plt.xlabel('Number of Documents')
    plt.ylabel('Source')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [f"{label} ({data[label][1]})" for label in labels]  # access the keyword string from the tuple
    plt.legend(handles, labels, title_fontsize='13', bbox_to_anchor=(-0.45, -0.15), loc='upper left', ncol=1,
               fontsize=6)

    plt.tight_layout()

    plt.gca().invert_yaxis()  # Highest ranked sources should be on top

    plt.show()


def compare_top_authors(data, N):
    """
    Compares the top N authors in terms of document count from multiple CSV files.

    Args:
    data (dict of pandas.DataFrame): A dictionary mapping CSV file paths to DataFrames of their data.
    N (int): The number of authors to display.

    Returns:
    None
    """

    # Empty DataFrame to hold the document counts for each author
    doc_counts_df = pd.DataFrame()

    all_author_counts = pd.Series(dtype=int)  # A series to hold document counts for all authors across CSVs

    for csv_path, df_keyword in data.items():
        df, _ = df_keyword  # unpack the tuple to separate the DataFrame and the keyword string

        # Expand the 'Authors' and 'Author(s) ID' columns into multiple columns and stack them into single columns
        authors = df['author_names'].str.split(';', expand=True).stack().str.strip()
        author_ids = df['author_ids'].str.split(';', expand=True).stack().str.strip()

        # Combine the author names and IDs into a single Series, removing any empty names or IDs
        combined = authors + ' [' + author_ids + ']'
        combined = combined[combined.str.strip() != '[]']

        # Filter out entries with no author name or ID available
        combined = combined[~combined.str.contains(r'\[No author id available\]', na=False)]
        combined = combined[~combined.str.contains(r'No author name available \[', na=False)]

        # Create a dictionary with author names as keys and their counts as values
        author_counts = combined.value_counts()

        all_author_counts = all_author_counts.add(author_counts, fill_value=0)  # Sum the counts of documents for each author

        # Add to the DataFrame
        doc_counts_df = doc_counts_df.join(author_counts.rename(csv_path), how='outer')

    # Select the top N authors
    top_authors = all_author_counts.nlargest(N).index

    # Filter the DataFrame to only include the top authors
   doc_counts_df = doc_counts_df.loc[top_authors]

    # Replace NaN values with zeros
    doc_counts_df = doc_counts_df.fillna(0)

    # Plot the data
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing
    plt.figure(figsize=(12, 6))  # Set the figure size

    print(doc_counts_df)

    # Get the 'Blues' colormap
    original_blues = plt.cm.get_cmap('Blues')

    # Create a new colormap that only uses the colors from 0.5 to 1.0 of the original colormap
    dark_blues = ListedColormap(original_blues(np.linspace(0.5, 1.0, 256)))

    # Before plotting, extract the author name from the index (remove the author ID)
    doc_counts_df.index = doc_counts_df.index.str.extract(r'(.*)(?=\s\[\d+\])', expand=False)
    doc_counts_df.plot(kind='barh', stacked=False, ax=plt.gca(), colormap=dark_blues)

    plt.title('Documents by Author')
    plt.xlabel('Number of Documents')
    plt.ylabel('Author')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [f"{label} ({data[label][1]})" for label in labels]  # access the keyword string from the tuple
    plt.legend(handles, labels, title_fontsize='13', bbox_to_anchor=(-0.1, -0.15), loc='upper left', ncol=1,
               borderaxespad=0., fontsize=6)

    plt.tight_layout()

    plt.gca().invert_yaxis()  # Highest ranked authors should be on top

    plt.show()


def list_additional_studies(data, original_data_name):
    """
    Lists the additional studies identified in new Scopus DataFrame compared to the original, and writes them to an
    Excel file.

    Args:
    data (dict): A dictionary where the keys are the filenames (representing the different queries) and the values are
    tuples containing the pandas DataFrame and the corresponding keyword string.
    original_data_name (str): The name of the original DataFrame against which to compare the others.

    Returns:
    None
    """

    # Ensure original_data_name exists in data
    if original_data_name not in data:
        raise ValueError(
            f"The original data name provided ({original_data_name}) does not exist in the data.")

    # Get the original DataFrame
    original_df = data[original_data_name][0]

    # Create a dictionary to hold the additional studies from each new DataFrame
   additional_studies = {}

    for query, df_tuple in data.items():
        # Skip if the current DataFrame is the original one
        if query == original_data_name:
            continue

        # Get the new DataFrame
        new_df = df_tuple[0]

        # Find the studies in the new DataFrame that are not in the original DataFrame
        additional_df = new_df[~new_df['eid'].isin(original_df['eid'])]

        # Add the additional studies to the dictionary
        # Keep all the columns
        additional_df = additional_df.copy()

        # Convert 'citedby_count' column to numeric (it might be stored as text) and sort in descending order
        additional_df['citedby_count'] = pd.to_numeric(additional_df['citedby_count'], errors='coerce').fillna(0)
        additional_df.sort_values('citedby_count', ascending=False, inplace=True)

        # Add to the dictionary
        additional_studies[query] = additional_df

    output_file = r'C:\Users\james\Desktop\additional_studies.xlsx'

    # Write the additional studies to an Excel file, with each DataFrame in a separate sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for query, additional_df in additional_studies.items():
            additional_df.to_excel(writer, sheet_name=query, index=False)

            # Get the openpyxl worksheet object
            worksheet = writer.sheets[query]

            # Set the width of the 'Title' column
            # 2350 pixels is approximately 279 Excel column width units
            # worksheet.column_dimensions['C'].width = 130  # 'C' corresponds to the 3rd column (Title)


directory = r'C:\Users\james\Desktop\comparisons'
keyword_filepath = r'C:\Users\james\Desktop\keywords_log.csv'

data = read_csv_files(directory, keyword_filepath)

# compare_total_studies(data)
# compare_cumulative_documents_by_year(data)
# compare_top_sources(data, 10)
# compare_top_authors(data, 10)
list_additional_studies(data, '1_LCA')


def main():
    search_log_manager = SearchLogManager()


if __name__ == '__main__':
    main()
