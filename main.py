from plots import *
from ppt_generation import generate_ppt_from_directory
from compare import *

# Options for plots
# plt.rcParams['figure.figsize'] = (10, 6)
# sbs.set('paper')
# docs = litstudy.sources.scopus.search_scopus("(LCA OR life cycle assessment OR life cycle analysis OR whole life carbon assessment) AND structur* AND (building* OR infrastructure)")
# litstudy.plot_year_histogram(docs, vertical=True);
# Load the CSV files
# docs1 = litstudy.load_csv('scopus.csv')
# print(docs1)
# docs = litstudy.search_scopus("(LCA OR life cycle assessment OR life cycle analysis OR whole life carbon assessment) AND structur* AND (building* OR infrastructure)")
# litstudy.plot_author_histogram(docs)

scopus_df = pd.read_csv(r'C:\Users\james\PycharmProjects\lcareview\assets\initial_scopus_csv\scopus.csv')

# plot_documents_by_year(scopus_df)
# plot_cumulative_documents_by_year(scopus_df)
# plot_documents_by_source(scopus_df, 15)
# plot_documents_by_year_and_source(scopus_df)
# plot_documents_by_country(scopus_df)
# plot_documents_by_continent(scopus_df)
# plot_documents_by_type(scopus_df)
# plot_documents_by_author(scopus_df)
# plot_top_cited_documents(scopus_df, 15)
# plot_abstracts_wordcloud(scopus_df)
# plot_titles_wordcloud(scopus_df)
# plot_index_keywords_wordcloud(scopus_df)
#
# keywords1 = ['life cycle assessment', 'life cycle analysis', 'embodied', 'whole life carbon']
# keywords2 = ['structure', 'structural', 'structural system']
# keywords3 = ['building', 'infrastructure']
# keywords4 = ['bridge', 'road', 'railway', 'airport', ' mine ', ' dam ', ' port ', 'harbor', 'sewage']
# plot_document_keyword_comparison(scopus_df, keywords1)
# plot_document_keyword_comparison(scopus_df, keywords2)
# plot_document_keyword_comparison(scopus_df, keywords3)
# plot_document_keyword_comparison(scopus_df, keywords4)

directory = r'C:\Users\james\PycharmProjects\lcareview\assets\comparison_scopus_csv'
keyword_filepath = r'C:\Users\james\PycharmProjects\lcareview\assets\keyword_strings.csv'
data = read_csv_files(directory, keyword_filepath)
# compare_total_studies(data)
# compare_cumulative_documents_by_year(data)
# compare_top_authors(data, 15)
# compare_top_sources(data, 15)
list_additional_studies(data, 'scopus_2')

generate_ppt_from_directory()