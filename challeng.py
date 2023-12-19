import os
import json
import requests
import PyPDF2
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
import openai
from transformers import pipeline
from googlesearch import search
from bs4 import BeautifulSoup

class Question:
    """
    Class to prompt the user to ask a question, and choose the two insures they wish to compare.
    """
    def __init__(self, json_file_name: str="insures.json"):
        self.question = self.ask_question()
        self.insure_choices = []
        self.json_raw = {}
        self.json_file_name = json_file_name
        self.json_data = self._get_insures_data()
        #self.time_created = datetime.now()

    def ask_question(self):
        """function to prompt user to write their questions:"""
        ques = input("How can I help you compare funeral cover?  ") # validate the question
        return ques
    
    def choose_insure(self, number_insure_choices: int=2):
        """To prompt user to choose the insures and validate each entry"""
        keys = []
        for i in range(number_insure_choices):
            while True:
                input_value = input(f"What is your No. {i + 1} insure do you wish to choose? ")
                if not input_value.isdigit():
                    print(f"Please enter valid number not characters: choose number from 1 to {len(self.json_data)}")
                elif len(input_value) < 0 or len(input_value) > 1:
                    print(f"Please enter valid value not more than 2 digits: choose number from 1 to {len(self.json_data)}")
                elif int(input_value) not in range(1, len(self.json_data) + 1):
                    print(f"Please choose number from 1 to {len(self.json_data)}")
                elif input_value in keys:
                    print(f"Value already choosen: Please enter different value from the first one: choose number from 1 to {len(self.json_data)}")
                elif int(input_value) in range(1, len(self.json_data) + 1):
                    keys.append(input_value)
                    break


        self.insure_choices = [self.json_data[choice] for choice in keys]
    
    def _get_insures_data(self):
        """The function organizes data: i.e {"1": "Capitec", "2": "Old Mutual" etc}"""
        insures = self._get_insures_list(self.json_file_name)
        self.json_raw = insures
        temp_insure_list = list(insures.keys())
        final_dict = {str(i + 1): temp_insure_list[i] for i in range(len(temp_insure_list))}
        #print(final_list)
        return final_dict
    
    def _get_insures_list(self, file_name: str):
        """The function load the original json file an read it"""
        current_directly = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_directly, file_name)

        # open the json file
        if file_name in os.listdir(current_directly):
            with open(json_file_path, 'r') as json_file:
                json_insures = json.load(json_file)
            return json_insures
        else:
            print(f"json file is not found in the current directory: {current_directly}.\nWeb scrapper will be used instead")
            # call web scrapper object and function
            return
        
    def _is_question_valid(self, question: str): # the function will be used to validate the question input.
        if len(question) < 5 or len(question) > 300:
            return False
        return True
    
    def __str__(self):
        return self.question
    

class PolicySearch(ABC):
    """
    interface class that will be used for searching for an answer.
    The search can be either from documents, website, API etc. Thus,
    the subclasses will implement the search type.
    """
    @abstractmethod
    def search_policy(self, *args, **kwargs):
        """The function to be implemented"""
        pass

class DocumentSearch(PolicySearch):
    """Subclass to implement the search from pdf document"""

    def search_policy(self, *args, **kwargs):

        results = [] #results will be stored in this list.
        question = kwargs["question"]
        text_list_dict = kwargs["page_list_text"]

        if kwargs["search_method"] == "transformers": 
            qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
                
            for page in text_list_dict:
                page_text = page["text"]

                result = qa_pipeline({
                    "question": question,
                    "context": page_text
                })

                if result:
                    final_result = {"answer": result["answer"], "page": str(page["page_num"]), "url": page["url"], "score": result["score"]}
                    results.append(final_result)

            # sort the results into desc based on score, then take the best.
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            return sorted_results[:1]
        
        elif kwargs["search_method"] == "chatgpt":
            api_key = kwargs["chatgpt_api_key"]
            insure = kwargs["insure"]
            openai.api_key = api_key

            # prompt to ask chatgpt.
            # More work needs to be done on prompt - give training examples, and results format needed.
            prompt = f"Search the following question: {question} in the pdf text: {text_list_dict} for insurance funeral policy. Specify the page number (page_num) you got the information from."
            
            # chatgpt function to search text.
            chatgpt_resp = openai.completions.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.5,
                max_tokens=150
            )

            answer = chatgpt_resp['choices'][0]['text'].strip()
            results[insure] = answer
            return results

    def read_pdf_text_from_url(self, pdf_url: str, insure_name: str):
        """The function reads texts from the web using url provided"""
        response = requests.get(pdf_url)

        pdf_data_per_page = []

        with BytesIO(response.content) as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            text = ""
            for page_num in range(pdf_reader.numPages):
                pdf_data = {}
                page = pdf_reader.getPage(page_num)
                text += page.extractText()
                pdf_data["page_num"] = page_num
                if insure_name.lower() == "capitec": # this is to remove the sentence of the bottow page and append to the right place.
                    page_num_str = str(page_num + 1)
                    text_to_remove = f"\nCapitec Bank is an authorised financial service (FSP46669)  and registered credit provider (NCRCP13). Capitec Bank Limited Reg.  No.: 1980/003695/06  \nUnique Document No.: Template  / 801 / V1 2.0 - 14/11/2021 (ddmmccyy)  Page {page_num_str} of \n&quot"
                    if page_num > 0: # the text need to be appended as a last sentence in each page
                        full_text = page.extract_text()
                        full_text = self._remove_last_sentence(full_text, text_to_remove)
                        full_text += text_to_remove # the text need to be appended as a last sentence in each page
                    else:
                        full_text = page.extract_text()
                        full_text += text_to_remove
                else:
                    full_text = page.extract_text()

                pdf_data["text"] = full_text
                pdf_data["url"] = pdf_url
                pdf_data_per_page.append(pdf_data)

        return pdf_data_per_page, text #return text as tuple containg, list of jsons text(per page) and just full string of the document.
    
    def _remove_last_sentence(self, raw_text: str, sentence_to_remove: str):
        result = raw_text.replace(sentence_to_remove, '')
        return result

class WebScrapperSearch(PolicySearch):
    """Subclass to implement the search from web , using web scrapping"""

    def search_policy(self, *args, **kwargs):
        question = kwargs["question"]
        results = search(question, num_results=5)
        for url in results:
            response = requests.get(url=url)
            soup = BeautifulSoup(response.text, "html.parser")
            #print(soup)
            #answer = soup.find("p").get_text()
            #print(answer)

            # more code to be implemented
        return results


class APISearch(PolicySearch):
    """Subclass to implement the search from api call"""
    def search_policy(self, *args, **kwargs):
        pass  # implement something


if __name__ == "__main__":

    def run():

        # instatiate the Question object
        question_obj = Question()

        print("\n*****************\n")
        print("Choose from below list: \n")

        for key, value in question_obj.json_data.items():
            print(f"{key}: {value}")
        print("\n*****************\n")

        question_obj.choose_insure()

        search_results = []
        for insure in question_obj.insure_choices:
            url = question_obj.json_raw[insure]["url"] #validate url
            search_type = question_obj.json_raw[insure]["search_type"] # we will manually specify the search type in future

            if search_type == "document":
                document_search_obj = DocumentSearch()

                # if url is invalid for document search, notify user to put valid entry in the json file
                if url == "" or not str(url).split('/')[-1].endswith(".pdf"):
                    print("\n******** File Input Error *********\n")
                    print(f"Note that the url {url} for {insure} is not valid for pdf search. \nPlease fix the insure.json. Web scrapper will be used instead.")
                    print("\n******** END *********\n")
                    # use webscrapper object. code to be implemented
                    return
                
                text_dict, text_str = document_search_obj.read_pdf_text_from_url(url, insure)
                if text_dict:
                    #print(text_dict)
                    question_to_ask = question_obj.question
                    api_key = "put your api key" # when using chatgpt as a search method.
                    search_result = document_search_obj.search_policy(question=question_to_ask, page_list_text=text_dict, insure=insure, chatgpt_api_key=None, search_method = "transformers")
                    search_results.append({insure: search_result})
                    
            elif search_type == "web_scrapper":
                scrapper_obj = WebScrapperSearch()
                question_to_ask = f"{question_obj.question} in {insure}"
                scrapper_results = scrapper_obj.search_policy(question=question_to_ask)
                # More code to be implemented.
            else:
                print("Please input valid search type in the insure.json file.")
                return 

        #display results:
        print("\n*****************\n")
        print(f"Question: {question_obj}\n")
        for res in search_results:
            for key, val in res.items():
                answer = val[0]["answer"]
                page_num = val[0]["page"]
                page_link = val[0]["url"]
                print(f"\n{key}:\n{answer}\n\nThis is from page number {page_num} of the document that can be found on {page_link}.")
        print("\n********** END *******\n")


# run
run()