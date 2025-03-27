# Import libraries
import csv
import time
import warnings
import pandas as pd
import requests
from ollama import chat
from pydantic import BaseModel
warnings.filterwarnings('ignore')

# Header for SEC EDGAR requests
headers = {'User-Agent': 'Alex Reyes reyesale@umich.edu'}

def obtain_urls(year, quarter, form, total_forms, company_list):

    """
    This function obtains specified forms for provided companies from the SEC EDGAR Website

    Input descriptions:
    year: in format 20XX as an integer
    quarter: as a string in format "QTR#" where # is 1, 2, 3, or 4
    form: string that is either "8-K", "10-K" or "10-Q"
    total_forms: the number of total forms to parse as an integer
    company_list: is a dictionary where the key is the company's name as a string
                  and the value is a list of strings in the form ['CIK#', 'Ticker']

    Output will be a list of lists containing ['Form URL', 'Company Name', 'Company Ticker']
    """

    # Download SEC EDGAR master data file containing all company form submissions for specified year/quarter #
    master_data = requests.get(
        f'https://www.sec.gov/Archives/edgar/full-index/{year}/{quarter}/master.idx', stream=True,
        headers=headers).content

    # Decode the master data file and split it by lines
    master_data = master_data.decode("utf-8").split('\n')

    # create list to store form urls
    form_urls = []

    for name, CIK in company_list.items():
        print(f"Processing {name} for {form} urls.")

        # Iterate through the master data lines row by row to find lines that include
        # the companies provided in company_list as input to this function
        for line in master_data:

            # This will check we are only collecting the specified total # of forms
            # Stop if we have the requested amount of forms
            if len(form_urls) >= total_forms:
                break

            # This will select lines of the master file that contain names of the
            # companies listed in our input file company_list
            if (name.lower() in line.lower()) and (form in line):
                line_info = line.strip()
                line_split = line_info.split('|')

                # Obtain portion of url from line to build full url later on
                url = line_split[-1]

                url2 = url.replace("-", "").replace(".txt", "")
                htm_url = 'https://www.sec.gov/Archives/' + url

                # Request the form data and get the filename
                company_form_data = requests.get(htm_url, stream=True, headers=headers).text
                company_form_data = company_form_data.split('FILENAME>')
                company_form_data = company_form_data[1].split('\n')[0]

                # Generate the finalized .htm URL
                company_form_url = 'https://www.sec.gov/Archives/' + url2 + '/' + company_form_data

                # Request the 8-K filing page with except error handling
                try:
                    response = requests.get(company_form_url, headers=headers)

                    # If we find the above regex in the file, we can then extract the 8-K press release url
                    form_urls.append([company_form_url, name, CIK[1]])
                    print(f"Found press release URL for {name}: {company_form_url}")

                except requests.RequestException as e:
                    print(f"Error fetching {company_form_url}: {e}")
                    continue

        # Recheck once we have broken out of inner loop to avoid
        # looping through another company
        if len(form_urls) >= total_forms:
            break

    print(f"\nCollected {len(form_urls)} valid form {form} URLs.")

    # Return the collected press release URLs
    return form_urls

##################################################################################################################################################################################
##################################################################################################################################################################################

def llm_parser(documents):
    """
    This function takes a list containing lists [urls, stock names,tickers] as input and returns a list
    of llm response outputs for each company form url in the input list.
    """

    # Timing how long total llm parsing and output will take
    start_time = time.time()

    # Create a structured output class for the llm response
    class DocumentOutput(BaseModel):
        date_of_report: str
        new_product: str
        product_description: str

    # Create list of lists to store llm outputs. First list will be row names since we will export to CSV
    output_list = [["company_name", "stock_name", "filing_time", "new_product", "product_description"]]

    # Iterate through the urls obtained from obtain_urls function
    for line_info in documents:

        url = line_info[0]

        # format name for file downloaded from each url as url name
        file_content = requests.get(url, stream=True, headers=headers).text

        # Create prompt to provide as input to the llm
        prompt = (
                f"Read the following file content and provide the following details:\n"
                f"- Date of the report\n"
                f"- Name of new product mentioned\n"
                f"- Description of the new product\n\n"
                f"Content:\n{file_content}\n\n"
                f"IMPORTANT RULES:\n"
                f"1. The New Product Description MUST be summarized in 180 characters or less.\n"
                f"2. Generic product names such as 'New Product 1' or similar names are NOT considered new products. Only list products with specific names.\n"
                f"3. Acquiring another business or business unit is NOT considered a new product.\n"
                f"4. If no new products are mentioned, output the following:\n"
                f"   - New Product: none\n"
                f"   - Product Description: No new products mentioned.\n"
                f"5. Senior notes due or common stock are NOT considered new products. .\n"
                f"6. Do not provide reasoning, explanations, or your thinking in the output.\n"
            )


        # Store response from llm model
        # Available models are llama3.2:3b, deepseek-r1:14b, deepseek-r1:8b
        llm_response = chat(model='deepseek-r1:8b', messages=[
            {
               'role': 'user',
                'content': prompt,
                'temperature': 0.2,
                'top_k': 10
            }
        ],
                            format=DocumentOutput.model_json_schema(),
                            )

        # Format llm response in specified format from DocumentOutput class above
        doc_output = DocumentOutput.model_validate_json(llm_response.message.content)

        # Append llm response to output list that will be written to CSV file
        output_list.append([line_info[1], line_info[2], doc_output.date_of_report,
                            doc_output.new_product.replace("&amp;", "and"),
                            doc_output.product_description.replace("&amp;", "and")])

    # If any products or descriptions were incorrectly left blank, enter in
    # correct description and name as none. We skip the header row.
    for row in output_list[1:]:
        if not row[3].strip():
            row[3] = "none"
        if not row[4].strip():
            row[4] = "No new products mentioned."

    # calculate total time for llm chat responses
    elapsed_time = time.time() - start_time
    print(f"Time taken for the chat operation: {elapsed_time:.2f} seconds")

    return output_list

##################################################################################################################################################################################
##################################################################################################################################################################################

def write_to_csv(parser_results):

    # Write the results form the llm to parser_output.csv file
    with open('testoutput.csv', "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(parser_results)

##################################################################################################################################################################################
##################################################################################################################################################################################

if __name__ == "__main__":

    # Read in SPY500 company data to obtain names, ciks, and tickers
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    # Create dictionary of company names, ciks, and tickers
    spy_companies = {}
    for i in range(0, len(tickers)):
        spy_companies[str(tickers.iloc[i]['Security'])] = [str(tickers.iloc[i]["CIK"]), str(tickers.iloc[i]["Symbol"])]

    # Add leading 0s to CIKs per SEC website guidance
    for ticker in spy_companies:
        spy_companies[ticker][0] = spy_companies[ticker][0].zfill(10)

    # Running the functions on the dictionary of requested companies
    # In this case, we are using SPY500 companies

    # Available form requests: '8-K', '10-K', or '10-Q'
    url_list = obtain_urls(2024, "QTR4", "8-K", 100, spy_companies)
    results = llm_parser(url_list)
    write_to_csv(results)