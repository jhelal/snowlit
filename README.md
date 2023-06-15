# Snowlit

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you start, ensure you meet the following requirements:

- You have installed the latest version of [Python](https://www.python.org/downloads/)
- Obtain APIKey from Scopus and InstToken from [Scopus](https://pybliometrics.readthedocs.io/en/stable/access.html)

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
