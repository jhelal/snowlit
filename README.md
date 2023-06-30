# SnowLit

SnowLit is a research literature analysis tool designed to leverage Scopus data for the acquisition, analysis, and visualisation of metadata pertaining to research papers. This tool features forward and backward snowballing search functions, utilising references and citations from the research papers to facilitate the discovery of supplementary scholarly work. The efficiency and depth of SnowLit make it a robust solution for research literature analysis.


In essence, this package offers four main features:

 - **Metadata Extraction**: SnowLit enables the extraction of metadata from scientific documents available in Scopus, systematically aggregating the data into a .csv file.
 - **Data Visualisation**: Beyond simple metadata acquisition, SnowLit automatically generates a series of graphical visualisations that offer in-depth insights. These visualisations, which are provided both as individual .png files and collectively within a .ppt file, cover aspects such as the number of papers per year, papers categorised by source, type, affiliation, country, continents, and author. Additionally, it provides visual representations of the top cited papers, as well as word clouds for titles and abstracts.
 - **Backward Snowballing**: SnowLit allows for the compilation and extraction of metadata from the references of a given research paper, delivering the results in a .csv file for further investigation and use.
 - **Forward Snowballing**: Similarly, it provides a function for forward snowballing, allowing for the aggregation and extraction of metadata from the citations of a research paper, with the data conveniently formatted in a .csv file.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you start, ensure you meet the following requirements:

- You have installed the latest version of [Python](https://www.python.org/downloads/)
- Obtain APIKey and InstToken from [Scopus](https://pybliometrics.readthedocs.io/en/stable/access.html)

### Installing

1. Clone the repository:

   ```sh
   git clone https://github.com/jhelal/snowlit.git
   ```

2. Change the working directory:

   ```sh
   cd snowlit
   ```

3. Create a virtual environment and activate it:

   ```
   python3 -m venv env

   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

4. Install the requirements:

   ```sh
   pip install -r requirements.txt
   ```

### Running the application

1. Run `main.py` python file

   ```sh
   python3 main.py
   ```

2. Follow the instructions from the terminal

   Based on the options you choose:

   - Scopus query search results will be stored at `search_results/<QUERY_ID>_<QUERY_NAME>/csv/scoups_results.csv`
   - Forward snowball results will be stored at `search_results/<QUERY_ID>_<QUERY_NAME>/csv/forward_snowball_results.csv`
   - Backward snowball results will be stored at `search_results/<QUERY_ID>_<QUERY_NAME>/csv/backward_snowball_results.csv`
   - All the images of plot will be stored at `search_results/<QUERY_ID>_<QUERY_NAME>/plots`
   - PPT will be generated and stored at `search_results/<QUERY_ID>_<QUERY_NAME>/snowlit_plots.pptx`
   - Results log will be stored at `search_results/results_log.csv`

## Folder Structure

```py
.
├── README.md # Documentation for the project
├── assets # Contains static files and resources used in the project
│ └── template.pptx
├── main.py # Main file to run the application
├── plots.py # Helper utility file to generate plots
├── ppt_generation.py # Generates pptx from the generated plots
├── requirements.txt
├── scopus.py # Scopus API functions
└── utils.py
```

## Known Issues

See issues [here](https://github.com/jhelal/snowlit/issues)

## Found a Bug?

Create an issue [here](https://github.com/jhelal/snowlit/issues/new/choose)

## Contact

James Helal

jameshelal@gmail.com
