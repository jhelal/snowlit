import pandas as pd
from pybliometrics.scopus import AbstractRetrieval, ScopusSearch


def backward_snowballing(scopus_csv_file, output_csv_file):
    """
    Compiles a DataFrame of references from a Scopus CSV file and exports it to a CSV file.

    Args:
    scopus_csv_file (str): The path to the Scopus CSV file.
    output_csv_file (str): The path to the output CSV file.

    Returns:
    None
    """

    # Read the Scopus CSV file into a DataFrame
    scopus_df = pd.read_csv(scopus_csv_file)

    # Extract the DOIs into a list
    doi_list = scopus_df['DOI'].dropna().tolist()

    # Create an empty DataFrame for the references
    references_df = pd.DataFrame()

    for index, doi in enumerate(doi_list, start=1):
        # Retrieve the abstract
        try:
            ab = AbstractRetrieval(doi, id_type='doi', view='FULL')

            # Append the references to the DataFrame
            for reference in ab.references:
                reference_series = pd.Series(reference._asdict())
                reference_series['Origin_DOI'] = doi  # add the original DOI to the series
                reference_series['Origin_Index'] = index  # add the document index to the series
                references_df = pd.concat([references_df, reference_series.to_frame().T], ignore_index=True)
        except Exception as e:
            print(f"Error occurred for DOI {doi}: {e}")

    # Eliminate duplicate references based on 'id' and 'doi'
    references_df.drop_duplicates(subset=['id', 'doi'], keep='first', inplace=True)

    # Rearrange columns to make 'Document_Index' and 'Origin_DOI' the first and second columns respectively
    cols = list(references_df.columns)
    cols.insert(0, cols.pop(cols.index('Origin_DOI')))
    cols.insert(0, cols.pop(cols.index('Origin_Index')))
    references_df = references_df.loc[:, cols]

    # Export the references DataFrame to a CSV file
    references_df.to_csv(output_csv_file, index=False)


def forward_snowballing(input_csv_path, output_csv_path):
    # Load the original document data from the CSV file
    original_documents = pd.read_csv(input_csv_path)
    original_document_eids = set(original_documents["EID"])

    # Define the column names for the output DataFrame
    column_names = ["Origin_Index", "Origin_DOI", "position", "id", "eid", "doi", "pii", "pubmed_id", "title",
                    "subtype", "subtypeDescription", "creator", "afid", "affilname", "affiliation_city",
                    "affiliation_country", "author_count", "author_names", "author_ids", "author_afids", "coverDate",
                    "coverDisplayDate", "publicationName", "issn", "source_id", "eIssn", "aggregationType", "volume",
                    "issueIdentifier", "article_number", "pageRange", "description", "authkeywords", "citedby_count",
                    "openaccess", "freetoread", "freetoreadLabel", "fund_acr", "fund_no", "fund_sponsor"]

    # Create an empty DataFrame for the citing documents
    df = pd.DataFrame(columns=column_names)

    # Create a set to keep track of the EIDs that have already been processed
    processed_eids = set()

    # Process each original document
    for i, original_document_row in original_documents.iterrows():
        original_document_eid = original_document_row["EID"]
        print(f"Processing original document {i + 1}/{len(original_documents)}...")

        # Search for documents that cite the original document
        citing_docs_search = ScopusSearch(f"REF({original_document_eid})")
        citing_docs_eids = citing_docs_search.get_eids()

        # Process each citing document
        for j, citing_doc_eid in enumerate(citing_docs_eids):
            # Skip the citing document if it's already been processed or if it's in the original documents
            if citing_doc_eid in processed_eids or citing_doc_eid in original_document_eids:
                continue

            print(f"   Processing citing document {j + 1}/{len(citing_docs_eids)}...")

            citing_doc = AbstractRetrieval(citing_doc_eid)
            row = {attr: getattr(citing_doc, attr, None) for attr in column_names[4:]}

            row["Origin_Index"] = i + 1
            row["Origin_DOI"] = original_document_row["DOI"]
            row["position"] = j + 1
            row["id"] = citing_doc_eid

            # Append the citing document to the DataFrame
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

            # Add the citing document's EID to the set of processed EIDs
            processed_eids.add(citing_doc_eid)

    # Export the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)


# path of test scopus csv
scopus_csv_file_path = r'C:\Users\james\Desktop\testing.csv'
output_csv_file_path = r'C:\Users\james\Desktop\forward_snowball_output.csv'

forward_snowballing(scopus_csv_file_path, output_csv_file_path)
