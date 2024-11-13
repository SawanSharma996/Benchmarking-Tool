from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from .forms import UploadFileForm
from .models import CompanyInfo
import pandas as pd
import time
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_POST
import json
from django.shortcuts import get_object_or_404
from io import BytesIO
from django.db.models import Q
import uuid
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from django.template.loader import render_to_string
from xhtml2pdf import pisa
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
import logging
logger = logging.getLogger(__name__)


def is_captcha_page(driver):
    page_source = driver.page_source.lower()
    # Simple checks for common CAPTCHA providers
    if 'recaptcha' in page_source or 'please verify you are a human' in page_source:
        return True
    return False


def get_screenshot(url, company_name):
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--window-size=1920,1080')
        options.add_argument("--lang=en")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(3) 
        
        driver.execute_script("document.body.style.zoom='100%'") 
        if is_captcha_page(driver):
            print(f"CAPTCHA detected on {url}. Skipping screenshot.")
            driver.quit()
            return None
        # Save the screenshot
        screenshot = driver.get_screenshot_as_png()
        driver.quit()
        return screenshot
    except Exception as e:
        print(f"Error capturing screenshot for {url}: {e}")
        return None

genai.configure(api_key=settings.OPENAI_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro')


def scrape_website_content(url):

    try:  

        headers = {
            'User-Agent': 'BenchmarkingTool/1.0 (Contact: sawansharma996@gmail.com)'
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract relevant text from the page
        # This can include text from <p>, <div>, and other elements
        texts = soup.find_all(text=True)
        
        # Filter out unwanted elements, such as scripts, styles, etc.
        blacklist = [
            '[document]',
            'noscript',
            'header',
            'html',
            'meta',
            'head',
            'input',
            'script',
            'style',
            # Add other tags as needed
        ]
        
        output = ''
        for t in texts:
            if t.parent.name not in blacklist:
                content = t.strip()
                if content:
                    output += '{} '.format(content)
        
        # Optionally, limit the amount of text to prevent overloading the AI model
        max_length = 5000  # Adjust as needed
        output = output[:max_length]
        
        return output
    except Exception as e:
        print(f"Error scraping website {url}: {e}")
        return None


def get_processed_count(request):
    batch_id = request.session.get('batch_id')
    if batch_id:
        total_uploaded = CompanyInfo.objects.filter(batch_id=batch_id).count()
    else:
        total_uploaded = 0
    return JsonResponse({'total_uploaded': total_uploaded})

def normalize_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url  # Default to HTTPS
    return url

def company_detail(request, company_id):
    batch_id = request.session.get('batch_id')
    company = get_object_or_404(CompanyInfo, id=company_id, batch_id=batch_id)
    return render(request, 'company_detail.html', {'company': company})

def export_pdf(request):
    batch_id = request.session.get('batch_id')
    if batch_id:
        companies = CompanyInfo.objects.filter(batch_id=batch_id)
    else:
        return HttpResponse("No data available for export.")

    companies_data = []
    for company in companies:
        screenshot_base64 = ''
        if company.screenshot and os.path.exists(company.screenshot.path):
            with open(company.screenshot.path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                screenshot_base64 = f"data:image/png;base64,{encoded_string}"
        companies_data.append({
            'company_name': company.company_name,
            'website_address': company.website_address,
            # Include other fields as needed
            'screenshot_base64': screenshot_base64,
        })

    # Render HTML template with company data
    html_string = render_to_string('export_pdf.html', {'companies': companies_data})

    # Create a PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="company_screenshots.pdf"'

    pisa_status = pisa.CreatePDF(html_string, dest=response)

    if pisa_status.err:
        return HttpResponse('Error generating PDF', status=500)
    return response

def fetch_company_details(website, company_name, retries=5):
    try:
        details = {}

        normalized_url = normalize_url(website)

        screenshot = get_screenshot(normalized_url, company_name)
        if screenshot:
            screenshot_filename = f"screenshots/{company_name}_{uuid.uuid4()}.png"
            path = default_storage.save(screenshot_filename, ContentFile(screenshot))
            details['screenshot_path'] = path
            details['captcha_detected'] = False
        else:
            details['screenshot_path'] = None
            # If CAPTCHA was detected, set the flag
            details['captcha_detected'] = True
        
        # Fetch content from the website
        scraped_content = scrape_website_content(normalized_url)
        
        if not scraped_content:
            return {
                'products_services': 'Could not fetch content from the website.',
                'business_description': 'Could not fetch content from the website.',
                'part_of_group': 'Could not fetch content from the website.'
            }
        
        # Now use the AI model to summarize the scraped content
        
        # Business Description
        prompt_bd = f"Summarize the following company information and tell me about the in what business this company is involved in in 50 words also give answer in english also dont include any headers, bold or any special symbols give answers in paragraphs only:\n\n{scraped_content}"
        details['business_description'] = generate_summary(prompt_bd)
        
        # Products or Services
        prompt_ps = f"From the following content, extract details about the company's products or services in 50 words  give answer in english also dont include any headers, bold or any special symbols give answers in paragraphs only:\n\n{scraped_content}"
        details['products_services'] = generate_summary(prompt_ps)
        
        # Part of Group
        prompt_pg = f"Based on the following content, determine if the company is part of a multinational group. in english also dont include any headers, bold or any special symbols give answers in paragraphs only.\n\n{scraped_content}"
        details['part_of_group'] = generate_summary(prompt_pg)
        
        return details
    except Exception as e:
        return {
            'products_services': f"Error fetching details: {str(e)}",
            'business_description': f"Error fetching details: {str(e)}",
            'part_of_group': f"Error fetching details: {str(e)}",
            'screenshot_path': None,
            'captcha_detected': False
        }

def generate_summary(prompt, retries=3):
    for attempt in range(retries):
        try:
            # Replace this with your actual code to interact with Gemini
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                time.sleep(sleep_time)
            else:
                return f"Error generating summary: {str(e)}"
@login_required
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                
                # Process the Excel file
                df = pd.read_excel(file)
                selected_columns = df[['Company name Latin alphabet', 'Website address', 'Trade description (English)', 'Full overview', 'Products & services']]
                
                # Rename columns to match expectations
                selected_columns.columns = ['company_name', 'website_address', 'trade_description', 'full_overview', 'products_servicesD']
                
                # Drop rows where company_name is null or empty
                selected_columns = selected_columns.dropna(subset=['company_name'])
                selected_columns = selected_columns[selected_columns['company_name'].str.strip() != '']
                
                batch_id = uuid.uuid4()
                request.session['batch_id'] = str(batch_id)
                # Save the company info data associated with this profile
                for _, row in selected_columns.iterrows():
                    CompanyInfo.objects.create(
                        company_name=row['company_name'],
                        website_address=row['website_address'],
                        trade_description=row['trade_description'],
                        full_overview=row['full_overview'],
                        products_servicesD=row['products_servicesD'],
                        batch_id=batch_id
                    )
     
                # Redirect to profile list after successful upload
                return redirect('display_excel')
            except Exception as e:
                return HttpResponse(f"Error processing file: {str(e)}")
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})

# views.py
def display_excel(request):
    batch_id = request.session.get('batch_id')
    if batch_id:
        data = list(CompanyInfo.objects.filter(batch_id=batch_id).values())
        return render(request, 'display_excel.html', {'data': data})
    else:
        # No batch_id in session, redirect to upload page or show a message
        return redirect('upload_file')
    

@csrf_exempt
def fetch_additional_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        website = data.get('website')
        company_name = data.get('company_name')
        trade_description = data.get('trade_description')
        full_overview = data.get('full_overview')


        if not website:
            return JsonResponse({'error': 'Website address is required'}, status=400)

        company_details = fetch_company_details(website, company_name)

        # Retrieve batch_id from session
        batch_id = request.session.get('batch_id')

        # Find and update the existing CompanyInfo object
        company = CompanyInfo.objects.filter(
            company_name=company_name,
            batch_id=batch_id
        ).first()

        if company:
            company.website_address = website
            company.trade_description = trade_description
            company.full_overview = full_overview
            company.products_services = company_details.get('products_services', '')
            company.business_description = company_details.get('business_description', '')
            company.part_of_group = company_details.get('part_of_group', '')
            company.screenshot = company_details.get('screenshot_path', '')

            company.save()
        else:
            # Handle the case where the company is not found
            return JsonResponse({'error': 'Company not found'}, status=404)

        return JsonResponse(company_details)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def generate_related_keywords(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            functionality_select = data.get('functionality_select', '')
            functionality_reject = data.get('functionality_reject', '')
            products_services = data.get('products_services', '')
            
            related_keywords = {}
            
            # Generate keywords for functionality select
            if functionality_select:
                prompt_select = f"Generate 20 related keywords or phrases for the company functionality: {functionality_select}. Provide the result as a comma-separated list."
                response_select = model.generate_content(prompt_select)
                related_keywords['functionality'] = [kw.strip() for kw in response_select.text.split(',')]
            
            # Generate keywords for functionality reject
            if functionality_reject:
                prompt_reject = f"Generate 20 related keywords or phrases for the company functionality: {functionality_reject}. Provide the result as a comma-separated list."
                response_reject = model.generate_content(prompt_reject)
                related_keywords['functionality_reject'] = [kw.strip() for kw in response_reject.text.split(',')]

            # Generate keywords for products/services
            if products_services:
                prompt_products = f"Generate 20 related keywords or phrases for the products/services: {products_services}. Provide the result as a comma-separated list."
                response_products = model.generate_content(prompt_products)
                related_keywords['products_services'] = [kw.strip() for kw in response_products.text.split(',')]
            
            return JsonResponse(related_keywords)
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word not in punctuation]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def expand_keywords(keywords):
    expanded = set()
    for keyword in keywords:
        # Add the original keyword
        expanded.add(keyword)
        # Get synonyms from WordNet
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                expanded.add(lemma.name())
    return list(expanded)

def prepare_corpus(companies, keywords):
    corpus = []
    for company in companies:
        text = company['description']  # Replace with the appropriate field
        tokens = preprocess_text(text)
        corpus.append(' '.join(tokens))
    # Preprocess keywords
    keyword_texts = [' '.join(preprocess_text(kw)) for kw in keywords]
    corpus.extend(keyword_texts)
    return corpus

def compute_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def compute_similarity(tfidf_matrix, num_companies, num_keywords):
    # Company vectors are the first num_companies entries
    company_vectors = tfidf_matrix[:num_companies]
    # Keyword vectors are the last num_keywords entries
    keyword_vectors = tfidf_matrix[num_companies:]
    # Compute cosine similarity between companies and keywords
    similarity_matrix = cosine_similarity(company_vectors, keyword_vectors)
    return similarity_matrix

@login_required
def get_past_uploads(request):
    # Assuming `batch_id` is stored in the session
    batch_id = request.session.get('batch_id')
    user = request.user
    if batch_id:
        # Filter CompanyInfo objects by batch_id
        past_uploads = CompanyInfo.objects.filter(user=user, batch_id=batch_id).values(
            'company_name', 'website_address', 'trade_description', 'products_services', 'full_overview'
        )
    else:
       
        # Optionally, you could return all uploads for the user if no batch_id is in the session
        past_uploads = CompanyInfo.objects.filter(user=user).values(
            'company_name', 'website_address', 'trade_description', 'products_services', 'full_overview'
        )

    return JsonResponse({'past_uploads': list(past_uploads)})


@csrf_exempt
@require_POST
def search_companies(request):
    try:
        data = json.loads(request.body)
        functionality_keywords = data.get('functionality_keywords', [])
        reject_functionality_keywords = data.get('reject_functionality_keywords', [])
        products_services_keywords = data.get('products_services_keywords', [])
        part_of_group = data.get('part_of_group', '').lower()

        # Expand keywords
        functionality_keywords_expanded = expand_keywords(functionality_keywords)
        reject_functionality_keywords_expanded = expand_keywords(reject_functionality_keywords)
        products_services_keywords_expanded = expand_keywords(products_services_keywords)

        batch_id = request.session.get('batch_id')
        if not batch_id:
            return JsonResponse({'error': 'Batch ID not found in session'}, status=400)

        companies = CompanyInfo.objects.filter(batch_id=batch_id)
        company_descriptions = []
        company_products_services = []
        company_list = []

        # Check if keyword lists are empty
        if not functionality_keywords_expanded and not products_services_keywords_expanded:
            return JsonResponse({'error': 'Please provide at least one functionality or product/service keyword.'}, status=400)

        for company in companies:
            # Get company description and products/services
            description = company.business_description or ''
            if description.strip() == "Could not fetch content from the website.":
                description = company.full_overview or ''
            description = ' '.join(preprocess_text(description))

            products_services = company.products_services or ''
            if products_services.strip() == "Could not fetch content from the website.":
                products_services = company.products_servicesD or ''
            products_services = ' '.join(preprocess_text(products_services))

            company_descriptions.append(description)
            company_products_services.append(products_services)
            company_list.append(company)

        results = []

        # Process functionality matching if keywords are provided
        if functionality_keywords_expanded:
            # Prepare corpus
            functionality_corpus = company_descriptions + functionality_keywords_expanded

            # Compute TF-IDF matrices
            tfidf_matrix_func, vectorizer_func = compute_tfidf_matrix(functionality_corpus)

            num_companies = len(company_list)
            num_func_keywords = len(functionality_keywords_expanded)

            # Compute similarities
            similarity_matrix_func = compute_similarity(tfidf_matrix_func, num_companies, num_func_keywords)
        else:
            similarity_matrix_func = None

        # Process products/services matching if keywords are provided
        if products_services_keywords_expanded:
            # Prepare corpus
            products_corpus = company_products_services + products_services_keywords_expanded

            # Compute TF-IDF matrices
            tfidf_matrix_prod, vectorizer_prod = compute_tfidf_matrix(products_corpus)

            num_companies = len(company_list)
            num_prod_keywords = len(products_services_keywords_expanded)

            # Compute similarities
            similarity_matrix_prod = compute_similarity(tfidf_matrix_prod, num_companies, num_prod_keywords)
        else:
            similarity_matrix_prod = None

        for idx, company in enumerate(company_list):
            # Functionality Matching
            if similarity_matrix_func is not None:
                if similarity_matrix_func.shape[1] == 0:
                    max_similarity_func = 0.0
                else:
                    max_similarity_func = similarity_matrix_func[idx].max()
                functionality_match = bool(max_similarity_func >= 0.07)  # Convert to native bool
                functionality_reason = f"Max similarity score: {max_similarity_func:.2f}"
            else:
                # If no functionality keywords provided, decide how to handle
                functionality_match = True  # or False, based on your logic
                functionality_reason = "No functionality keywords provided."

            # Reject Functionality Matching
            reject_match = any(keyword in company_descriptions[idx] for keyword in reject_functionality_keywords_expanded)
            reject_reason = "Reject functionality match found." if reject_match else "No reject functionality match found."

            # Products/Services Matching
            if similarity_matrix_prod is not None:
                if similarity_matrix_prod.shape[1] == 0:
                    max_similarity_prod = 0.0
                else:
                    max_similarity_prod = similarity_matrix_prod[idx].max()
                product_match = bool(max_similarity_prod >= 0.07)  # Convert to native bool
                product_reason = f"Max similarity score: {max_similarity_prod:.2f}"
            else:
                # If no products/services keywords provided, decide how to handle
                product_match = True  # or False, based on your logic
                product_reason = "No products/services keywords provided."

            # Group Matching Logic (same as before)
            company_group = (company.part_of_group or '').lower()
            if part_of_group == 'yes':
                group_match = 'yes' in company_group
                group_reason = "Company is part of a group." if group_match else "Company is not part of a group."
            elif part_of_group == 'no':
                group_match = 'no' in company_group or not company_group
                group_reason = "Company is not part of a group." if group_match else "Company is part of a group."
            else:
                group_match = True
                group_reason = "No group preference specified."

            # Determine Accept/Reject based on matches
            if reject_match or not (product_match and functionality_match and group_match):
                accept_status = 'Reject'
            else:
                accept_status = 'Accept'

            # Compile result
            results.append({
                'company_name': company.company_name,
                'accept_status': accept_status,
                'product_match': product_match,
                'product_reason': product_reason,
                'functionality_match': functionality_match,
                'functionality_reason': functionality_reason,
                'reject_match': reject_match,
                'reject_reason': reject_reason,
                'group_match': group_match,
                'group_reason': group_reason,
            })

        # Store results in session
        request.session['search_results'] = results

        # Return redirect URL in JSON response
        return JsonResponse({'redirect_url': '/search_results/'})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
    
def search_results(request):
    results = request.session.get('search_results', [])
    return render(request, 'search_results.html', {'results': results})

def export_search_results(request):
    try:
        batch_id = request.session.get('batch_id')
        if not batch_id:
            return HttpResponse("No data available for export.")

        # First Sheet: Company Information from display_excel.html
        companies = CompanyInfo.objects.filter(batch_id=batch_id)
        companies_values = companies.values()

        if not companies_values.exists():
            return HttpResponse("No company data available for export.")

        # Convert QuerySet to DataFrame and define column order and labels
        companies_df = pd.DataFrame.from_records(companies_values)
        companies_columns = {
            'company_name': 'Company Name',
            'website_address': 'Website Address',
            'trade_description': 'Trade Description',
            'full_overview': 'Full Overview',
            'products_servicesD': 'Products/Services (Database)',
            'business_description': 'Business Description (By API)',
            'products_services': 'Products/Services (By API)',
            'part_of_group': 'Part of Group (By API)',
        }
        # Select and rename columns
        companies_df = companies_df[list(companies_columns.keys())].rename(columns=companies_columns)

        # Second Sheet: Accept/Reject Results from search_results
        search_results = request.session.get('search_results', [])
        if not search_results:
            return HttpResponse("No search results available for export.")

        results_df = pd.DataFrame(search_results)
        results_columns = {
            'company_name': 'Company Name',
            'accept_status': 'Accept/Reject',
            'product_match': 'Product Match',
            'product_reason': 'Product Reason',
            'functionality_match': 'Functionality Match',
            'functionality_reason': 'Functionality Reason',
            'group_match': 'Group Match',
            'group_reason': 'Group Reason',
        }
        # Select and rename columns
        results_df = results_df[list(results_columns.keys())].rename(columns=results_columns)

        # Create a BytesIO buffer to save the Excel file
        buffer = BytesIO()

        # Use pandas to save the DataFrames as an Excel file with two sheets
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write the dataframes without headers (we'll write headers separately to apply formatting)
            companies_df.to_excel(writer, index=False, sheet_name='Company Information', header=False, startrow=1)
            results_df.to_excel(writer, index=False, sheet_name='Search Results', header=False, startrow=1)

            # Access the workbook and worksheets
            workbook = writer.book
            company_sheet = writer.sheets['Company Information']
            results_sheet = writer.sheets['Search Results']

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            wrapped_text_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

            # Format Company Information sheet
            for col_num, value in enumerate(companies_df.columns.values):
                company_sheet.write(0, col_num, value, header_format)
                # Adjust column width
                column_len = companies_df[value].astype(str).map(len).max()
                column_len = max(column_len, len(value)) + 2  # Adding extra space

                if value in ['Full Overview', 'Products/Services (Database)', 'Business Description (By API)', 'Products/Services (By API)']:
                    # Apply text wrapping to columns with long text
                    company_sheet.set_column(col_num, col_num, column_len, wrapped_text_format)
                else:
                    company_sheet.set_column(col_num, col_num, column_len)

            # Freeze the top row
            company_sheet.freeze_panes(1, 0)
            # Add filters
            company_sheet.autofilter(0, 0, companies_df.shape[0], companies_df.shape[1] - 1)

            # Format Search Results sheet
            for col_num, value in enumerate(results_df.columns.values):
                results_sheet.write(0, col_num, value, header_format)
                # Adjust column width
                column_len = results_df[value].astype(str).map(len).max()
                column_len = max(column_len, len(value)) + 2

                if value in ['Product Reason', 'Functionality Reason', 'Group Reason']:
                    # Apply text wrapping to columns with long text
                    results_sheet.set_column(col_num, col_num, column_len, wrapped_text_format)
                else:
                    results_sheet.set_column(col_num, col_num, column_len)

            # Freeze the top row
            results_sheet.freeze_panes(1, 0)
            # Add filters
            results_sheet.autofilter(0, 0, results_df.shape[0], results_df.shape[1] - 1)

        # Set up the HTTP response
        response = HttpResponse(
            buffer.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename=search_results.xlsx'

        return response
    except Exception as e:
        # Log the exception
        return HttpResponse(f"An error occurred while exporting data: {str(e)}")


def export_excel(request):
    batch_id = request.session.get('batch_id')
    if batch_id:
        companies = CompanyInfo.objects.filter(batch_id=batch_id)
    else:
        # No batch_id in session, handle accordingly
        return HttpResponse("No data available for export.")
    
    # Create a DataFrame from the queryset
    df = pd.DataFrame.from_records(companies.values())
    
    # Create a BytesIO buffer to save the Excel file
    buffer = BytesIO()
    
    # Use pandas to save the DataFrame as an Excel file
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Companies')
    
    # Set up the HTTP response
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=companies_data.xlsx'
    
    # Write the Excel file to the response
    response.write(buffer.getvalue())
    
    return response
