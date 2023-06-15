import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pycountry_convert as pc
import pycountry
from wordcloud import WordCloud

from utils import save_plot_as_image


def plot_total_documents(df):
    """
    Reads the DataFrame of a Scopus data and plots a bar chart of total documents.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    total_publications = len(df)  # total number of documents

    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    # Create a larger figure to make the bar thinner
    # plt.figure(figsize=(3, 6))

    # Create a bar plot with 'blue' color
    ax = sns.barplot(x=["Total Publications"], y=[total_publications], color="#276196")

    plt.title("Total Number of Documents")
    plt.ylabel("Number of Documents")

    # Add the total number of documents above the bar
    ax.text(0, total_publications, total_publications, color="black", ha="center")

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    save_plot_as_image(plt, "total_documents.png")


def plot_documents_by_year(df):
    """
    Reads the list of namedtuple records of a Scopus data and plots a bar chart of documents by year using seaborn.

    Args:
    records (list of namedtuple): List of records of the Scopus data.

    Returns:
    None
    """

    # Extract the year from the 'coverDate' column
    df["Year"] = df["coverDate"].apply(lambda x: int(x.split("-")[0]))

    # Count the number of documents per year
    doc_counts = df["Year"].value_counts().sort_index()

    # Create a list of all years from the earliest in your data to the latest
    all_years = np.arange(df["Year"].min(), df["Year"].max() + 1)

    # Reindex your data to include all years, filling missing years with 0
    doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

    # Plot the data
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    # Create a bar plot with 'blue' color
    ax = sns.barplot(
        x=doc_counts_reindexed.index, y=doc_counts_reindexed.values, color="#276196"
    )

    plt.title("Documents by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Documents")

    # Display every nth label to avoid overlapping
    n = 2  # change this value to get more or less labels
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)

    # Rotate the labels to prevent overlap
    plt.xticks(rotation=45)

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    save_plot_as_image(plt, "documents_per_year.png")


def plot_cumulative_documents_by_year(df):
    """
    Reads the list of namedtuples and plots an area chart of cumulative documents by year.

    Args:
    data (list of namedtuples): The list of namedtuples returned by ScopusSearch.

    Returns:
    None
    """

    # Convert the 'coverDate' to datetime format
    df["coverDate"] = pd.to_datetime(df["coverDate"])

    # Extract the year and create a new 'Year' column
    df["Year"] = df["coverDate"].dt.year

    # Count the number of documents per year
    doc_counts = df["Year"].value_counts().sort_index()

    # Create a list of all years from the earliest in your data to the latest
    all_years = np.arange(df["Year"].min(), df["Year"].max() + 1)

    # Reindex your data to include all years, filling missing years with 0
    doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

    # Calculate the cumulative sum of the documents
    cumulative_doc_counts = doc_counts_reindexed.cumsum()

    # Plot the data
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    plt.fill_between(
        cumulative_doc_counts.index,
        cumulative_doc_counts.values,
        color="#276196",
        alpha=0.8,
    )
    plt.plot(
        cumulative_doc_counts.index,
        cumulative_doc_counts.values,
        color="#276196",
        alpha=0.6,
    )

    plt.title("Cumulative Documents by Year")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Number of Documents")

    # Set the limits of the x-axis and y-axis
    plt.xlim(df["Year"].min(), df["Year"].max())
    plt.ylim(0, cumulative_doc_counts.max() + 5)

    # Rotate the labels to prevent overlap
    plt.xticks(rotation=45)

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'cumulative_documents_by_year.png')

    save_plot_as_image(plt, "cumulative_documents_by_year.png")


def plot_documents_by_source(df, N=10):
    """
    Reads the dataframe of Scopus data and plots a bar chart of documents by source.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): The number of sources to display.

    Returns:
    None
    """

    # Filter to include only the top N sources based on document count
    top_sources = df["publicationName"].value_counts().nlargest(N).index
    df_filtered = df[df["publicationName"].isin(top_sources)]

    # Count the number of documents per source
    doc_counts = df_filtered["publicationName"].value_counts()

    # Convert the series to a DataFrame and display the results
    doc_counts_df = doc_counts.to_frame().reset_index()
    doc_counts_df.columns = ["publicationName", "Number of Documents"]

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient="h"
    )

    for i, val in enumerate(doc_counts.values):
        barplot.text(val + 1, i, val, va="center")

    plt.title("Documents by Source")
    plt.xlabel("Number of Documents")
    plt.ylabel("Source")

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_source.png')

    save_plot_as_image(plt, "documents_by_source.png")


def plot_documents_by_year_and_source(df, N=10):
    """
    Reads the dataframe of a Scopus CSV file and plots a stacked area chart of documents by year and source.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): The number of sources to display.

    Returns:
    None
    """

    # Create a new column 'Year' by extracting the year from the 'coverDate' column
    df["Year"] = pd.to_datetime(df["coverDate"]).dt.year

    # Filter to include only the top 10 sources based on document count
    top_sources = df["publicationName"].value_counts().nlargest(N).index
    df_filtered = df[df["publicationName"].isin(top_sources)]

    # Count the number of documents per year and source
    doc_counts = (
        df_filtered.groupby(["Year", "publicationName"]).size().unstack(fill_value=0)
    )

    # Reindex the DataFrame to include all years from the earliest to the latest
    all_years = np.arange(df_filtered["Year"].min(), df_filtered["Year"].max() + 1)
    doc_counts_reindexed = doc_counts.reindex(all_years, fill_value=0)

    # Plot the data using a stacked area chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    plt.stackplot(
        doc_counts_reindexed.index,
        doc_counts_reindexed.T.values,
        labels=doc_counts_reindexed.columns,
        alpha=0.75,
    )

    plt.title("Documents by Year and Source")
    plt.xlabel("Year")
    plt.ylabel("Number of Documents")
    plt.legend(loc="upper left")

    # Set the x-axis limits to the min and max years
    plt.xlim(df_filtered["Year"].min(), df_filtered["Year"].max())

    # Rotate the labels to prevent overlap
    plt.xticks(rotation=45)

    # Set y-axis label to display as integer
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0f}".format(y)))

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_year_and_source.png')

    save_plot_as_image(plt, "documents_by_year_and_source.png")


def plot_documents_by_affiliation(df, N=10):
    """
    Reads the dataframe of a Scopus CSV file and plots a bar chart of documents by affiliation.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): number of affiliations to display

    Returns:
    None
    """

    # Drop rows where 'affilname' is NaN
    df = df.dropna(subset=["affilname"]).copy()

    # Create a unique identifier for each document
    df["DocumentID"] = df["eid"]

    # Split the 'affilname' column into a list of affiliations, then explode the list into separate rows
    df["affilname"] = df["affilname"].str.split(";")
    df = df.explode("affilname")

    # Clean up any extra spaces in the affiliations
    df["affilname"] = df["affilname"].str.strip()

    # Drop duplicates based on DocumentID and Affiliation
    df = df.drop_duplicates(subset=["DocumentID", "affilname"])

    # Filter to include only the top N affiliations based on document count
    top_affiliations = df["affilname"].value_counts().nlargest(N).index
    df_filtered = df[df["affilname"].isin(top_affiliations)]

    # Count the number of documents per affiliation
    doc_counts = df_filtered["affilname"].value_counts()

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient="h"
    )

    for i, val in enumerate(doc_counts.values):
        barplot.text(
            val + 0.1, i, val, va="center"
        )  # Reduced label position adjustment

    plt.title("Documents by Affiliation")
    plt.xlabel("Number of Documents")
    plt.ylabel("Affiliation")

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_affiliation.png')

    save_plot_as_image(plt, "documents_by_affiliation.png")


def plot_documents_by_country(df, N=10):
    """
    Reads the dataframe of a Scopus CSV file and plots a bar chart of documents by country.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): number of countries to display

    Returns:
    None
    """

    # Drop rows where 'affiliation_country' is NaN
    df = df.dropna(subset=["affiliation_country"]).copy()

    # Create a unique identifier for each document
    df["DocumentID"] = df["creator"] + df["title"] + df["coverDate"].astype(str)

    # Split the 'affiliation_country' column into a list of countries, then explode the list into separate rows
    df["affiliation_country"] = df["affiliation_country"].str.split(";")
    df = df.explode("affiliation_country")

    # Clean up any extra spaces in the countries
    df["affiliation_country"] = df["affiliation_country"].str.strip()

    # Drop duplicates based on DocumentID and Country
    df = df.drop_duplicates(subset=["DocumentID", "affiliation_country"])

    # Filter to include only the top N countries based on document count
    top_countries = df["affiliation_country"].value_counts().nlargest(N).index
    df_filtered = df[df["affiliation_country"].isin(top_countries)]

    # Count the number of documents per country
    doc_counts = df_filtered["affiliation_country"].value_counts()

    # Convert the series to a DataFrame and display the results
    doc_counts_df = doc_counts.to_frame().reset_index()
    doc_counts_df.columns = ["Country", "Number of Documents"]

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient="h"
    )

    for i, val in enumerate(doc_counts.values):
        barplot.text(
            val + 0.1, i, val, va="center"
        )  # Reduced label position adjustment

    plt.title("Documents by Country")
    plt.xlabel("Number of Documents")
    plt.ylabel("Country")

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_country.png')

    save_plot_as_image(plt, "documents_by_country.png")


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
        "AF": "Africa",
        "AS": "Asia",
        "EU": "Europe",
        "NA": "North America",
        "OC": "Oceania",
        "SA": "South America",
        "AN": "Antarctica",
    }

    # Drop rows where 'affiliation_country' is NaN
    df = df.dropna(subset=["affiliation_country"]).copy()

    # Create a unique identifier for each document
    df["DocumentID"] = df["eid"]

    # Split the 'affiliation_country' column into a list of countries, then explode the list into separate rows
    df["affiliation_country"] = df["affiliation_country"].str.split(";")
    df = df.explode("affiliation_country")

    # Clean up any extra spaces in the affiliations
    df["affiliation_country"] = df["affiliation_country"].str.strip()

    # Convert from country name to continent name
    df["Continent"] = df["affiliation_country"].apply(
        lambda x: continent_name.get(
            pc.country_alpha2_to_continent_code(pycountry.countries.get(name=x).alpha_2)
        )
        if pycountry.countries.get(name=x)
        else None
    )

    # Drop duplicates based on DocumentID and Continent
    df = df.drop_duplicates(subset=["DocumentID", "Continent"])

    # Count the number of documents per continent
    doc_counts = df["Continent"].value_counts()

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        y=doc_counts.index, x=doc_counts.values, palette="Blues_r", orient="h"
    )

    for i, val in enumerate(doc_counts.values):
        barplot.text(
            val + 0.3, i, val, va="center"
        )  # Reduced label position adjustment

    plt.title("Documents by Continent")
    plt.xlabel("Number of Documents")
    plt.ylabel("Continent")

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_continent.png')

    save_plot_as_image(plt, "documents_by_continent.png")


def plot_documents_by_type(df):
    """
    Reads the dataframe of Scopus CSV file and plots a vertical bar chart of documents by type.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Count the number of documents per type
    doc_counts = df["subtypeDescription"].value_counts()

    # Convert the series to a DataFrame and display the results
    doc_counts_df = doc_counts.to_frame().reset_index()
    doc_counts_df.columns = ["Document Type", "Number of Documents"]

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        x=doc_counts.values, y=doc_counts.index, palette="Blues_r", orient="h"
    )

    for i, val in enumerate(doc_counts.values):
        barplot.text(
            val + 0.3, i, val, va="center"
        )  # Reduced label position adjustment

    plt.title("Documents by Type")
    plt.xlabel("Number of Documents")
    plt.ylabel("Document Type")

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_type.png')

    save_plot_as_image(plt, "documents_by_type.png")


def plot_documents_by_author(df, N=10):
    """
    Reads the dataframe of Scopus CSV file and plots a horizontal bar chart of documents by author.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): number of authors to display

    Returns:
    None
    """

    # Drop rows where 'author_names' or 'author_ids' are NaN
    df = df.dropna(subset=["author_names", "author_ids"])

    # Expand the 'author_names' and 'author_ids' columns into multiple columns and stack them into single columns
    authors = df["author_names"].str.split(";", expand=True).stack()
    author_ids = df["author_ids"].str.split(";", expand=True).stack()

    # Remove leading/trailing whitespace
    authors = authors.str.strip()
    author_ids = author_ids.str.strip()

    # Combine the author names and IDs into a single Series, removing any empty names or IDs
    combined = authors + " [" + author_ids + "]"
    combined = combined[combined.str.strip() != "[]"]

    # Create a dictionary with author names as keys and their counts as values
    author_counts = combined.value_counts().nlargest(N)
    author_names = author_counts.index.str.extract(r"(.*) \[")[
        0
    ]  # extract author names from the index

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing

    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        x=author_counts.values, y=author_names, palette="Blues_r", orient="h"
    )

    for i, val in enumerate(author_counts.values):
        barplot.text(
            val + 0.1, i, val, va="center"
        )  # Reduced label position adjustment

    plt.title("Documents by Author")
    plt.xlabel("Number of Documents")
    plt.ylabel("Author")

    plt.tight_layout()

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'documents_by_author.png')

    save_plot_as_image(plt, "documents_by_author.png")


def plot_abstracts_wordcloud(df):
    """
    Reads the dataframe of Scopus CSV file and generates a high-resolution word cloud from the Abstract column.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Drop rows where 'description' is NaN
    df = df.dropna(subset=["description"])

    # Join all abstracts into a single string
    text = " ".join(df["description"])

    # Create the word cloud with higher resolution
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white", width=800, height=800
    ).generate(text)

    # Display the generated image
    plt.figure(figsize=[10, 5])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save the figure to a PNG file
    # save_plot_as_image(plt, 'abstracts_wordcloud.png')

    save_plot_as_image(plt, "abstracts_wordcloud.png")


def plot_titles_wordcloud(df):
    """
    Reads the dataframe of a Scopus CSV file and generates a high-resolution word cloud from the Title column.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Drop rows where 'Title' is NaN
    df = df.dropna(subset=["title"])

    # Join all titles into a single string
    text = " ".join(df["title"])

    # Create the word cloud with higher resolution
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white", width=400, height=400
    ).generate(text)

    # Display the generated image
    plt.figure(figsize=[10, 5])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save the figure to a PNG file
    save_plot_as_image(plt, "titles_wordcloud.png")


def plot_top_cited_documents(df, N=10):
    """
    Reads the dataframe of a Scopus CSV file and generates a bar chart of the top N cited documents. It also prints a
    DataFrame of these documents with their titles and other important metadata.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    N (int): The number of top-cited documents to display.

    Returns:
    None
    """

    df["coverDate"] = pd.to_datetime(df["coverDate"])

    # Extract the year and create a new 'Year' column
    df["Year"] = df["coverDate"].dt.year

    # Fill NA values in 'Cited by' with 0
    df["Cited by"] = df["citedby_count"].fillna(0)

    # Sort by 'Cited by' in descending order and take the top N rows
    df_top_cited = df.sort_values("Cited by", ascending=False).head(N)

    # Create a new column 'Citation Label' for the labels on the x-axis
    df_top_cited["Citation Label"] = (
        df_top_cited["author_names"].apply(
            lambda x: x.split(", ")[0].split(" ")[0] + " et al."
            if ", " in x
            else x.split(" ")[0]
        )
        + " ("
        + df_top_cited["Year"].astype(str)
        + ")"
    )

    # Display the DataFrame of top-cited documents with their titles and other important metadata
    # print(df_top_cited[["Citation Label", "Title", "Cited by"]])

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing
    plt.figure(figsize=(10, 6))  # Set the figure size

    # barplot = sns.barplot(
    #     x="Citation Label", y="Cited by", data=df_top_cited, palette="Blues_r"
    # )

    plt.title("Top " + str(N) + " Cited Documents")
    plt.xlabel("Document")
    plt.ylabel("Number of Citations")
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align them with bars

    plt.tight_layout()

    # Save the figure to a PNG file
    save_plot_as_image(plt, "top_cited_documents.png")

    # plt.show()


# -------------- Below functions are not used in the project --------------


def plot_index_keywords_wordcloud(df):
    """
    Reads the dataframe of Scopus CSV file and generates a high-resolution word cloud from the Index Keywords column.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.

    Returns:
    None
    """

    # Drop rows where 'Index Keywords' is NaN
    df = df.dropna(subset=["Index Keywords"])

    # Split the 'Index Keywords' column into a list of keywords, then join them into a single string
    df["Index Keywords"] = df["Index Keywords"].str.split(";")
    text = " ".join([" ".join(keywords) for keywords in df["Index Keywords"]])

    # Create the word cloud with higher resolution
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white", width=400, height=400
    ).generate(text)

    # Display the generated image
    plt.figure(figsize=[10, 5])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save the figure to a PNG file
    save_plot_as_image(plt, "index_keywords_wordcloud.png")

    # plt.show()


def plot_keyword_comparison(df, keywords):
    """
    Reads the dataframe of a Scopus CSV file and counts the number of times each keyword is mentioned in the Titles,
    Abstracts, and Keywords.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    keywords (list): A list of keywords to search for.

    Returns:
    None
    """

    # Convert to lowercase for case-insensitive matching
    df = df.apply(lambda x: x.astype(str).str.lower())

    # Initialize a dictionary to store the counts for each keyword
    keyword_counts = {keyword: 0 for keyword in keywords}

    # Iterate through each keyword and count the number of occurrences in the Titles, Abstracts, and Keywords
    for keyword in keywords:
        keyword_counts[keyword] += df["Title"].str.contains(keyword).sum()
        keyword_counts[keyword] += df["Abstract"].str.contains(keyword).sum()
        keyword_counts[keyword] += df["Index Keywords"].str.contains(keyword).sum()

    # Convert the dictionary to a pandas Series for easier plotting
    keyword_counts_series = pd.Series(keyword_counts)

    # Sort the series in descending order
    keyword_counts_series = keyword_counts_series.sort_values(ascending=False)

    # Convert the series to a DataFrame and display the results
    keyword_counts_df = keyword_counts_series.to_frame().reset_index()
    keyword_counts_df.columns = ["Keyword", "Number of Occurrences"]
    print(keyword_counts_df)

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing
    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        x=keyword_counts_series.values, y=keyword_counts_series.index, palette="Blues_r"
    )

    for i, val in enumerate(keyword_counts_series.values):
        barplot.text(val + 1, i, val, va="center")

    plt.title("Keyword Comparison")
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Keyword")

    plt.tight_layout()

    # Save the figure to a PNG file
    save_plot_as_image(plt, "keyword_comparison.png")

    # plt.show()


def plot_document_keyword_comparison(df, keywords):
    """
    Reads the dataframe of Scopus CSV file and counts the number of documents that mention each keyword in the Titles,
    Abstracts, and Keywords.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    keywords (list): A list of keywords to search for.

    Returns:
    None
    """

    # Convert to lowercase for case-insensitive matching
    df = df.apply(lambda x: x.astype(str).str.lower())

    # Initialize a dictionary to store the counts for each keyword
    keyword_counts = {keyword: 0 for keyword in keywords}

    # Iterate through each keyword and count the number of documents that mention it
    for keyword in keywords:
        for _, row in df.iterrows():
            if (
                keyword in row["Title"]
                or keyword in row["Abstract"]
                or keyword in row["Index Keywords"]
            ):
                keyword_counts[keyword] += 1

    # Convert the dictionary to a pandas Series for easier plotting
    keyword_counts_series = pd.Series(keyword_counts)

    # Sort the series in descending order
    keyword_counts_series = keyword_counts_series.sort_values(ascending=False)

    # Convert the series to a DataFrame and display the results
    keyword_counts_df = keyword_counts_series.to_frame().reset_index()
    keyword_counts_df.columns = ["Keyword", "Number of Documents"]
    print(keyword_counts_df)

    # Plot the data using a bar chart
    sns.set_style(
        "whitegrid"
    )  # Set the seaborn style to "whitegrid" for easier viewing
    plt.figure(figsize=(10, 6))  # Set the figure size

    barplot = sns.barplot(
        x=keyword_counts_series.values, y=keyword_counts_series.index, palette="Blues_r"
    )

    for i, val in enumerate(keyword_counts_series.values):
        barplot.text(val + 1, i, val, va="center")

    plt.title("Document Keyword Comparison")
    plt.xlabel("Number of Documents")
    plt.ylabel("Keyword")

    plt.tight_layout()

    # Save the figure to a PNG file
    save_plot_as_image(plt, "document_keyword_comparison.png")

    # plt.show()


def plot_document_keyword_cooccurrence_matrix(df, keywords):
    """
    Reads the dataframe of a Scopus CSV file and generates a co-occurrence matrix for the given keywords. Each cell in
    the matrix represents the number of documents in which a pair of keywords co-occur either in the Title, Abstract,
    or Index Keywords fields.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    keywords (list): A list of keywords to search for.

    Returns:
    None
    """

    # Convert to lowercase for case-insensitive matching
    df = df.apply(lambda x: x.astype(str).str.lower())

    # Initialize a DataFrame to store the co-occurrence counts for each keyword pair
    cooccurrence_df = pd.DataFrame(index=keywords, columns=keywords)
    cooccurrence_df = cooccurrence_df.fillna(0)  # fill initial NaN values with 0

    # Iterate through each pair of keywords and count the number of documents where both appear
    for keyword1 in keywords:
        for keyword2 in keywords:
            if keyword1 != keyword2:  # exclude counting the keyword with itself
                cooccurrence_count = (
                    (
                        df["Title"].str.contains(keyword1)
                        | df["Abstract"].str.contains(keyword1)
                        | df["Index Keywords"].str.contains(keyword1)
                    )
                    & (
                        df["Title"].str.contains(keyword2)
                        | df["Abstract"].str.contains(keyword2)
                        | df["Index Keywords"].str.contains(keyword2)
                    )
                ).sum()
                cooccurrence_df.at[keyword1, keyword2] = cooccurrence_count

    # Suppress scientific notation
    pd.set_option("display.float_format", lambda x: "%.0f" % x)

    # Plot the co-occurrence matrix using a heatmap
    plt.figure(figsize=(10, 8))  # adjust as needed
    sns.heatmap(cooccurrence_df, annot=True, fmt="d", cmap="YlGnBu")

    plt.title("Keyword Co-occurrence Matrix")
    plt.xlabel("Keyword")
    plt.ylabel("Keyword")
    plt.tight_layout()  # adjusts subplot params so that the subplot fits in to the figure area

    # Save the figure to a PNG file
    save_plot_as_image(plt, "documents_keyword_cooccurence_matrix.png")

    # plt.show()

    # Reset float format to default
    pd.reset_option("display.float_format")


def plot_topics_lda(df, num_topics):
    """
    Reads the dataframe of a Scopus CSV file and applies LDA topic modeling to the abstracts of the papers. It then
    displays a bar chart showing the number of documents per topic.

    Args:
    df (pandas.DataFrame): DataFrame of the Scopus data.
    num_topics (int): The number of topics to extract.

    Returns:
    None
    """

    # Preprocess the abstracts
    stop_words = set(stopwords.words("english"))
    abstracts = (
        df["Abstract"]
        .fillna("")
        .apply(
            lambda x: [
                word
                for word in word_tokenize(x.lower())
                if word.isalpha() and word not in stop_words
            ]
        )
    )

    # Create a dictionary representation of the abstracts, and convert to bag-of-words
    dictionary = corpora.Dictionary(abstracts)
    corpus = [dictionary.doc2bow(abstract) for abstract in abstracts]

    # Apply LDA
    lda_model = gensim.models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=15
    )

    # Print the topics
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)

    # Assigns the topics to the documents in corpus
    lda_corpus = lda_model[corpus]

    # Find the dominant topic for each sentence
    dominant_topics = []
    for doc in lda_corpus:
        doc_topics = sorted(doc, key=lambda x: x[1], reverse=True)
        dominant_topic = doc_topics[0][0]
        dominant_topics.append(dominant_topic)

    df["Dominant Topic"] = dominant_topics

    # Prepare the topics for the x-axis labels
    topic_labels = {
        n: ", ".join(re.findall(r'"(.*?)"', t[1])) for n, t in enumerate(topics)
    }

    # Replace the topic numbers with the topic labels in the DataFrame
    df["Dominant Topic"] = df["Dominant Topic"].map(topic_labels)

    # Display the number of documents by dominant topic
    topic_counts = df["Dominant Topic"].value_counts()

    plt.figure(figsize=(10, 6))
    topic_counts.plot(kind="bar", color="skyblue")
    plt.title("Number of Abstracts by Dominant Topic")
    plt.xlabel("Topic")
    plt.ylabel("Number of Abstracts")
    plt.xticks(rotation=45, ha="right")

    # Adjust the plot layout to make sure nothing is cropped
    plt.tight_layout()

    plt.show()
