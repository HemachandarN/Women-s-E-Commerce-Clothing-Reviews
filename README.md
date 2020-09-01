# Women's E-Commerce-Clothing-Reviews
Analyzing Customer reviews using Text Mining to classify whether they will recommend a product or not. 

## **Abstract**

Customer feedback is a crucial source of information describing user experience with a company and its service. Just like in product development, efficient use of feedback can help identify and prioritize opportunities for company’s further development.

This readme describes how the code for our Text Analysis and Classification Model works.

## **Obejective** 

In our analysis, we will utilize the power of text mining to do an in-depth analysis of customer reviews on an e-commerce clothing site data and build a classification model to predict whether the customer will recommend the product or not.

It will help retailers to have an understanding about their products, mistakes and customer satisfaction.


## **Methodology** 

<img src = "https://github.com/pinkesh-nayak/job_aggregator/blob/master/data/process.PNG">

## **Data Source**<br>
### **Kaggle**<br>
Link: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews<br>

## **Libraries Used**
import pandas as pd<br>
import numpy as np<br>
from scipy import stats<br>
import string<br>
import seaborn as sns<br>
import matplotlib.pyplot as plt<br>
import matplotlib.gridspec as gridspec<br>
from sklearn.model_selection import train_test_split as split<br>
from sklearn import metrics<br> 
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve<br>
import nltk<br>
from nltk.corpus import stopwords<br>
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer <br>
from nltk.stem import PorterStemmer, LancasterStemmer<br>
from sklearn.feature_extraction.text import CountVectorizer<br>
from sklearn.feature_extraction.text import TfidfTransformer<br>
from nltk.tokenize import word_tokenize<br>
from nltk.probability import FreqDist<br>
import spacy<br>
import re<br>
from wordcloud import WordCloud<br>
from imblearn.over_sampling import SMOTE<br>
from nltk.stem.snowball import SnowballStemmer<br>
from nltk.sentiment.vader import SentimentIntensityAnalyzer<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.linear_model import LogisticRegression<br>
from sklearn.naive_bayes import GaussianNB<br>
from sklearn import svm<br>
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score<br>
import warnings<br>
warnings.filterwarnings('ignore') <br>
from IPython.display import Image<br>
%matplotlib inline<br>

## **Data Acquistion and Preprocessing** 

### **Randstand: Web Scraping**

1. scrapeDataFromPage(page, writer) | Function to fetch required data from a page for a particular job.<br>
2. fetchDataFromRandstad() | Function to get link of every jobs present on https://www.randstad.com/jobs/united-states/q-data-science/ calling above function scrapeDataFromPage() <br> 
3. fetchDataFromRandstadRSS() | To perform scraping once hourly from the RSS feeds<br>

### **Adzuna API** 

1. fetchDataFromAdzuna() | Checking for all the number of pages present in API and fetching all job listings from each page. <br>
2. fetchDataFromAdzuna(1) | It will fetch all data from previous day till today from the API.<br>

### **GitHub API**

1. fetchDataFromGithub() | Checking for all the number of pages present and fetching all job listings from each page.<br>

### **The Muse API**

1. fetchDataFromMuse() | Checking for all the number of pages present and fetching all job listings from each page.<br>

### **United States Citizenship and Immigration Services (USCIS) H1B data** 

1. fetchDataFromUSCIS() | We are fetching 2019 H-1B Employer Data which is openly available on USCIS Website.<br>

### **Merging Job data and USCIS data**

1. mergeWithUSCISData(company) | Merges jobs data with USCIS data to get the H1B statistics for each company in jobs data. <br>

### **Exporting the final results to a CSV file**

### **Accessing Final Job data**

1. searchForJobs(title, location, job_posted) | Function for the User to query for jobs. <br>


## File Sizes
Dated on : 06/14/2019

 File Name | Size |
 --- | --- |
 randstad.csv 							                   | 0.99 MB
 adzuna.csv 					           | 11.4 MB
 github.csv 		        | 256 KB
 muse.csv 					               	   	| 124 KB
 uscis.csv 					               	   	| 3.63 MB
 job_database.csv 					               	   	| 13 MB


## Column information of the final merge file
Columns | Definition |
 --- | --- |
 company  							                   |Name of the company
 title 					           |Job Title
 location 		        |Location of the posting
 link  					               	   	|Link to apply
 job_posted							                	      |Date posted 
 description  					                	|Job description 
 catergory 								                     |Job Category
 source 					           |Source from whether the job was fetched
 initial_approvals_2019 			       |H-1B petitions with “New employment” or “New concurrent employment” whose first decision is an approval.
 initial_denial_2019 							                  |H-1B petitions with “New employment” or “New concurrent employment” whose first decision is an denial.
 continuing_approvals_2019							                      	|H-1B petitions with anything other than “New employment” or “New concurrent employment” whose first decision is approval.
 continuing_denials_2019					                	|H-1B petitions with anything other than “New employment” or “New concurrent employment” whose first decision is denial.

## **How to Access the above final_job data file** <br>
searchForJobs(title, location, job_posted) | Function for the User to query for jobs.Filters availabe: title: Job title, location: Job location, and job_posted: Date of job posting will return all the records from that date to current date.<br>

## **Data Directories** <br>
1. './data/ranstad/randstad.csv' | Randstad Job's Data
2. './data/adzuna/adzuna.csv' | Adzuna Job's Data
3. './data/github/github.csv' | Github Job's Data
4. './data/muse/muse.csv' | The Muse Job's Data
5. './data/uscis/uscis.csv' | USCIS HIB Data
6. './data/job_database/job_database.csv' | Final Job Data

## **Who might be interested in this data**

1. Job seekers, majorly international applicants. 
2. Job Board Applications.
3. Educational institutes.
4. Data scientists who want to perform further analysis.  
5. Consultancies. 
6. Goverment agencies to generate various employement related surveys. 

## **Challenges Faced**
1. Since different source different job features, some source which are present in one data source may not be availaable in the other one. <br> 
2. Company name present in jobs data may not match with USCIS Employer name since it may have been registered with a different name. We are using 'contains()' function. <br>
The quality of the data can be improved by web scrapping from multiple sources of websites. But most of the sites don’t allow scrapping. We need to get permission. <br> 
3. Many APIs have rate limit which limits the data we can access from that site.<br> 



