from pybliometrics.scopus import AbstractRetrieval

from itertools import combinations
import networkx as nx
from pybliometrics.scopus import ScopusSearch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import namedtuple
import pandas as pd
import pycountry_convert as pc
import pycountry
from wordcloud import WordCloud

query = '(TITLE-ABS-KEY((LCA OR life cycle assessment OR life cycle analysis OR whole life carbon assessment OR ' \
        'embodied) AND structur* AND (building* OR infrastructure)))'
#
s = ScopusSearch(query)

# Get the results as a list of namedtuples
results = s.results

# Create a pandas DataFrame from your results
df = pd.DataFrame(results)
#
# Save the DataFrame as a CSV
df.to_csv("scopus_results.csv", index=False)




def read_csv_as_namedtuple(csv_path):
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Get the column names from the DataFrame
        cols = df.columns.tolist()

        # Define a namedtuple type with the column names as fields
        Record = namedtuple('Record', cols)

        # Use a list comprehension to convert each row of the DataFrame into a namedtuple and store them in a list
        records = [Record(*row) for row in df.itertuples(index=False)]

        return records

# # Use the function
# csv_path = 'scopus_results.csv'
# records = read_csv_as_namedtuple(csv_path)
# # Convert the list of namedtuples to a DataFrame
# scopus_df = pd.DataFrame(records)
# print(scopus_df.columns)


def plot_total_documents(df):
    """
    Reads the DataFrame of a Scopus data and plots a bar chart of total documents.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    total_publications = len(df)  # total number of documents

    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    # Create a larger figure to make the bar thinner
    # plt.figure(figsize=(3, 6))

    # Create a bar plot with 'blue' color
    ax = sns.barplot(x=["Total Publications"], y=[total_publications], color='#276196')

    plt.title('Total Number of Documents')
    plt.ylabel('Number of Documents')

    # Add the total number of documents above the bar
    ax.text(0, total_publications, total_publications, color='black', ha="center")

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    plt.show()


def plot_documents_by_year(df):
    """
    Reads the list of namedtuple records of a Scopus data and plots a bar chart of documents by year using seaborn.

    Args:
    records (list of namedtuple): List of records of the Scopus data.

    Returns:
    None
    """

    # Extract the year from the 'coverDate' column
    df['Year'] = df['coverDate'].apply(lambda x: int(x.split('-')[0]))

    # Count the number of documents per year
    doc_counts = df['Year'].value_counts().sort_index()

    # Create a list of all years from the earliest in your data to the latest
    all_years = np.arange(df['Year'].min(), df['Year'].max() + 1)

    # Reindex your data to include all years, filling missing years with 0
    doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

    # Plot the data
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    # Create a bar plot with 'blue' color
    ax = sns.barplot(x=doc_counts_reindexed.index, y=doc_counts_reindexed.values, color='#276196')

    plt.title('Documents by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Documents')

    # Display every nth label to avoid overlapping
    n = 2  # change this value to get more or less labels
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)

    # Rotate the labels to prevent overlap
    plt.xticks(rotation=45)

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    plt.show()


def plot_cumulative_documents_by_year(df):
    """
    Reads the list of namedtuples and plots an area chart of cumulative documents by year.

    Args:
    data (list of namedtuples): The list of namedtuples returned by ScopusSearch.

    Returns:
    None
    """

    # Convert the 'coverDate' to datetime format
    df['coverDate'] = pd.to_datetime(df['coverDate'])

    # Extract the year and create a new 'Year' column
    df['Year'] = df['coverDate'].dt.year

    # Count the number of documents per year
    doc_counts = df['Year'].value_counts().sort_index()

    # Create a list of all years from the earliest in your data to the latest
    all_years = np.arange(df['Year'].min(), df['Year'].max() + 1)

    # Reindex your data to include all years, filling missing years with 0
    doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

    # Calculate the cumulative sum of the documents
    cumulative_doc_counts = doc_counts_reindexed.cumsum()

    # Plot the data
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    plt.fill_between(cumulative_doc_counts.index, cumulative_doc_counts.values, color="#276196", alpha=0.8)
    plt.plot(cumulative_doc_counts.index, cumulative_doc_counts.values, color="#276196", alpha=0.6)

    plt.title('Cumulative Documents by Year')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Number of Documents')

    # Set the limits of the x-axis and y-axis
    plt.xlim(df['Year'].min(), df['Year'].max())
    plt.ylim(0, cumulative_doc_counts.max() + 5)

    # Rotate the labels to prevent overlap
    plt.xticks(rotation=45)

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'cumulative_documents_by_year.png')

    plt.show()


def plot_documents_by_source(df, N):
    """
    Reads the dataframe of Scopus data and plots a bar chart of documents by source.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): The number of sources to display.

    Returns:
    None
    """

    # Filter to include only the top N sources based on document count
    top_sources = df['publicationName'].value_counts().nlargest(N).index
    df_filtered = df[df['publicationName'].isin(top_sources)]

    # Count the number of documents per source
    doc_counts = df_filtered['publicationName'].value_counts()

    # Convert the series to a DataFrame and display the results
    doc_counts_df = doc_counts.to_frame().reset_index()
    doc_counts_df.columns = ['publicationName', 'Number of Documents']
    print(doc_counts_df)

    # Plot the data using a bar chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient='h')

    for i, val in enumerate(doc_counts.values):
        barplot.text(val + 1, i, val, va='center')

    plt.title('Documents by Source')
    plt.xlabel('Number of Documents')
    plt.ylabel('Source')

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_source.png')

    plt.show()


def plot_documents_by_year_and_source(df, N):
    """
    Reads the dataframe of a Scopus CSV file and plots a stacked area chart of documents by year and source.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): The number of sources to display.

    Returns:
    None
    """

    # Create a new column 'Year' by extracting the year from the 'coverDate' column
    df['Year'] = pd.to_datetime(df['coverDate']).dt.year

    # Filter to include only the top 10 sources based on document count
    top_sources = df['publicationName'].value_counts().nlargest(N).index
    df_filtered = df[df['publicationName'].isin(top_sources)]

    # Count the number of documents per year and source
    doc_counts = df_filtered.groupby(['Year', 'publicationName']).size().unstack(fill_value=0)

    # Reindex the DataFrame to include all years from the earliest to the latest
    all_years = np.arange(df_filtered['Year'].min(), df_filtered['Year'].max() + 1)
    doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

    # Plot the data using a stacked area chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    plt.stackplot(doc_counts_reindexed.index, doc_counts_reindexed.T.values, labels=doc_counts_reindexed.columns,
                  alpha=0.75)

    plt.title('Documents by Year and Source')
    plt.xlabel('Year')
    plt.ylabel('Number of Documents')
    plt.legend(loc='upper left')

    # Set the x-axis limits to the min and max years
    plt.xlim(df_filtered['Year'].min(), df_filtered['Year'].max())

    # Rotate the labels to prevent overlap
    plt.xticks(rotation=45)

    # Set y-axis label to display as integer
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_year_and_source.png')

    plt.show()


def plot_documents_by_affiliation(df, N):
    """
    Reads the dataframe of a Scopus CSV file and plots a bar chart of documents by affiliation.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): number of affiliations to display

    Returns:
    None
    """

    # Drop rows where 'affilname' is NaN
    df = df.dropna(subset=['affilname']).copy()

    # Create a unique identifier for each document
    df['DocumentID'] = df['eid']

    # Split the 'affilname' column into a list of affiliations, then explode the list into separate rows
    df['affilname'] = df['affilname'].str.split(';')
    df = df.explode('affilname')

    # Clean up any extra spaces in the affiliations
    df['affilname'] = df['affilname'].str.strip()

    # Drop duplicates based on DocumentID and Affiliation
    df = df.drop_duplicates(subset=['DocumentID', 'affilname'])

    # Filter to include only the top N affiliations based on document count
    top_affiliations = df['affilname'].value_counts().nlargest(N).index
    df_filtered = df[df['affilname'].isin(top_affiliations)]

    # Count the number of documents per affiliation
    doc_counts = df_filtered['affilname'].value_counts()

    # Plot the data using a bar chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient='h')

    for i, val in enumerate(doc_counts.values):
        barplot.text(val + 0.1, i, val, va='center')  # Reduced label position adjustment

    plt.title('Documents by Affiliation')
    plt.xlabel('Number of Documents')
    plt.ylabel('Affiliation')

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_affiliation.png')

    plt.show()


def plot_documents_by_country(df, N):
    """
    Reads the dataframe of a Scopus CSV file and plots a bar chart of documents by country.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): number of countries to display

    Returns:
    None
    """

    # Drop rows where 'affiliation_country' is NaN
    df = df.dropna(subset=['affiliation_country']).copy()

    # Create a unique identifier for each document
    df['DocumentID'] = df['creator'] + df['title'] + df['coverDate'].astype(str)

    # Split the 'affiliation_country' column into a list of countries, then explode the list into separate rows
    df['affiliation_country'] = df['affiliation_country'].str.split(';')
    df = df.explode('affiliation_country')

    # Clean up any extra spaces in the countries
    df['affiliation_country'] = df['affiliation_country'].str.strip()

    # Drop duplicates based on DocumentID and Country
    df = df.drop_duplicates(subset=['DocumentID', 'affiliation_country'])

    # Filter to include only the top N countries based on document count
    top_countries = df['affiliation_country'].value_counts().nlargest(N).index
    df_filtered = df[df['affiliation_country'].isin(top_countries)]

    # Count the number of documents per country
    doc_counts = df_filtered['affiliation_country'].value_counts()

    # Convert the series to a DataFrame and display the results
    doc_counts_df = doc_counts.to_frame().reset_index()
    doc_counts_df.columns = ['Country', 'Number of Documents']

    # Plot the data using a bar chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient='h')

    for i, val in enumerate(doc_counts.values):
        barplot.text(val + 0.1, i, val, va='center')  # Reduced label position adjustment

    plt.title('Documents by Country')
    plt.xlabel('Number of Documents')
    plt.ylabel('Country')

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_country.png')

    plt.show()


def plot_documents_by_continent(df):
    """
    Reads the dataframe of a Scopus CSV file and plots a bar chart of documents by continent.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Define the dictionary mapping from continent code to continent name
    continent_name = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'OC': 'Oceania',
        'SA': 'South America',
        'AN': 'Antarctica'
    }

    # Drop rows where 'affiliation_country' is NaN
    df = df.dropna(subset=['affiliation_country']).copy()

    # Create a unique identifier for each document
    df['DocumentID'] = df['eid']

    # Split the 'affiliation_country' column into a list of countries, then explode the list into separate rows
    df['affiliation_country'] = df['affiliation_country'].str.split(';')
    df = df.explode('affiliation_country')

    # Clean up any extra spaces in the affiliations
    df['affiliation_country'] = df['affiliation_country'].str.strip()

    # Convert from country name to continent name
    df['Continent'] = df['affiliation_country'].apply(lambda x: continent_name.get(
        pc.country_alpha2_to_continent_code(pycountry.countries.get(name=x).alpha_2)) if pycountry.countries.get(
        name=x) else None)

    # Drop duplicates based on DocumentID and Continent
    df = df.drop_duplicates(subset=['DocumentID', 'Continent'])

    # Count the number of documents per continent
    doc_counts = df['Continent'].value_counts()

    # Plot the data using a bar chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient='h')

    for i, val in enumerate(doc_counts.values):
        barplot.text(val + 0.3, i, val, va='center')  # Reduced label position adjustment

    plt.title('Documents by Continent')
    plt.xlabel('Number of Documents')
    plt.ylabel('Continent')

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_continent.png')

    plt.show()


def plot_documents_by_type(df):
    """
    Reads the dataframe of Scopus CSV file and plots a vertical bar chart of documents by type.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Count the number of documents per type
    doc_counts = df['subtypeDescription'].value_counts()

    # Convert the series to a DataFrame and display the results
    doc_counts_df = doc_counts.to_frame().reset_index()
    doc_counts_df.columns = ['Document Type', 'Number of Documents']

    # Plot the data using a bar chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(x=doc_counts.values, y=doc_counts.index, palette="Blues_r", orient='h')

    for i, val in enumerate(doc_counts.values):
        barplot.text(val + 0.3, i, val, va='center')  # Reduced label position adjustment

    plt.title('Documents by Type')
    plt.xlabel('Number of Documents')
    plt.ylabel('Document Type')

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_type.png')

    plt.show()


def plot_documents_by_author(df, N):
    """
    Reads the dataframe of Scopus CSV file and plots a horizontal bar chart of documents by author.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): number of authors to display

    Returns:
    None
    """

    # Drop rows where 'author_names' or 'author_ids' are NaN
    df = df.dropna(subset=['author_names', 'author_ids'])

    # Expand the 'author_names' and 'author_ids' columns into multiple columns and stack them into single columns
    authors = df['author_names'].str.split(';', expand=True).stack()
    author_ids = df['author_ids'].str.split(';', expand=True).stack()

    # Remove leading/trailing whitespace
    authors = authors.str.strip()
    author_ids = author_ids.str.strip()

    # Combine the author names and IDs into a single Series, removing any empty names or IDs
    combined = authors + ' [' + author_ids + ']'
    combined = combined[combined.str.strip() != '[]']

    # Create a dictionary with author names as keys and their counts as values
    author_counts = combined.value_counts().nlargest(N)
    author_names = author_counts.index.str.extract(r'(.*) \[')[0]  # extract author names from the index

    # Plot the data using a bar chart
    sns.set_style("whitegrid")  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(x=author_counts.values, y=author_names, palette="Blues_r", orient='h')

    for i, val in enumerate(author_counts.values):
        barplot.text(val + 0.1, i, val, va='center')  # Reduced label position adjustment

    plt.title('Documents by Author')
    plt.xlabel('Number of Documents')
    plt.ylabel('Author')

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_author.png')

    plt.show()


def plot_abstracts_wordcloud(df):
    """
    Reads the dataframe of Scopus CSV file and generates a high-resolution word cloud from the Abstract column.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Drop rows where 'description' is NaN
    df = df.dropna(subset=['description'])

    # Join all abstracts into a single string
    text = ' '.join(df['description'])

    # Create the word cloud with higher resolution
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", width=800, height=800).generate(text)

    # Display the generated image
    plt.figure(figsize=[10,5])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'abstracts_wordcloud.png')

    plt.show()




# plot_abstracts_wordcloud(scopus_df)