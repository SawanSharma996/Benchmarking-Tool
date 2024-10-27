from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from .forms import UploadFileForm
from .models import CompanyInfo
import pandas as pd
import time
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt
import json
from io import BytesIO
from django.db.models import Q
import uuid
from selenium.webdriver.chrome.options import Options
from selenium import webdriver


genai.configure(api_key=settings.OPENAI_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro')



def get_processed_count(request):
    total_uploaded = CompanyInfo.objects.count()
    return JsonResponse({'total_uploaded': total_uploaded})


def fetch_company_details(website, retries=5):
    try:
        details = {}
        
        # Fetch Products or Services
        prompt = f"Please draft products or service details from the official website of the company. The primary data source is the official website {website} of the company, which provides comprehensive and reliable information about the company's product or services. The overall response should be less than 50 words."
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                details['products_services'] = response.text
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    time.sleep(sleep_time)
                else:
                    details['products_services'] = f"Error fetching products/services: {str(e)}"

        # Fetch Business Description
        prompt = f'''Please draft:

- A brief business description from the official website {website} of the Company.

- Products or service details from the official website {website} of the Company.

- identify if the company is multinational group from the official website {website} of the Company

The primary data source is the official website of the company, which provides comprehensive and reliable information about the company's business, products, services, and corporate structure.'''
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                details['business_description'] = response.text
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    time.sleep(sleep_time)
                else:
                    details['business_description'] = f"Error fetching business description: {str(e)}"

        # Fetch Part of Group
        prompt = f"Please identify if the company is a multinational group. from the official website {website} of the company. The primary data source is the official website of the company, which provides comprehensive and reliable information about the company's offices, subsidiaries, joint ventures, and other associated members. The overall response should be less than 50 words."
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                details['part_of_group'] = response.text
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    time.sleep(sleep_time)
                else:
                    details['part_of_group'] = f"Error fetching part of group info: {str(e)}"

        return details
    except Exception as e:
        return {
            'products_services': f"Error fetching details: {str(e)}",
            'business_description': f"Error fetching details: {str(e)}",
            'part_of_group': False
        }

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                
                # Process the Excel file
                df = pd.read_excel(file)
                selected_columns = df[['Company name Latin alphabet', 'Website address', 'Trade description (English)', 'Full overview']]
                
                # Rename columns to match expectations
                selected_columns.columns = ['company_name', 'website_address', 'trade_description', 'full_overview']
                
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

        company_details = fetch_company_details(website)

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
            functionality = data.get('functionality', '')
            products_services = data.get('products_services', '')
            
            related_keywords = {}
            
            if functionality:
                prompt = f"Generate 10 related keywords or phrases for the company functionality: {functionality}. Provide the result as a comma-separated list."
                response = model.generate_content(prompt)
                related_keywords['functionality'] = [kw.strip() for kw in response.text.split(',')]
            
            if products_services:
                prompt = f"Generate 10 related keywords or phrases for the products/services: {products_services}. Provide the result as a comma-separated list."
                response = model.generate_content(prompt)
                related_keywords['products_services'] = [kw.strip() for kw in response.text.split(',')]
            
            return JsonResponse(related_keywords)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def search_companies(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            functionality_keywords = data.get('functionality_keywords', [])
            products_services_keywords = data.get('products_services_keywords', [])
            part_of_group = data.get('part_of_group', '').lower()

            batch_id = request.session.get('batch_id')
            companies = CompanyInfo.objects.filter(batch_id=batch_id)

            results = []

            for company in companies:
                # Initialize match flags and reasons
                product_match = False
                functionality_match = False
                group_match = False
                product_reason = ''
                functionality_reason = ''
                group_reason = ''

                # Check Products/Services match
                company_products_services = company.products_services or ''
                for keyword in products_services_keywords:
                    if keyword.lower() in company_products_services.lower():
                        product_match = True
                        product_reason = f"Matched keyword '{keyword}' in Products/Services."
                        break
                if not product_match:
                    product_reason = "No matching keywords found in Products/Services."

                # Check Functionality match
                company_functionality = company.business_description or ''
                for keyword in functionality_keywords:
                    if keyword.lower() in company_functionality.lower():
                        functionality_match = True
                        functionality_reason = f"Matched keyword '{keyword}' in Functionality."
                        break
                if not functionality_match:
                    functionality_reason = "No matching keywords found in Functionality."

                # Check Part of Group match
                company_group = (company.part_of_group or '').lower()
                if part_of_group:
                    if part_of_group in company_group:
                        group_match = True
                        group_reason = f"Company is part of group '{company.part_of_group}'."
                    else:
                        group_reason = f"Company is not part of the specified group '{part_of_group}'."
                else:
                    group_match = True  # If no group specified, consider it a match
                    group_reason = "No group specified; defaulted to match."

                # Determine Accept/Reject
                if product_match and functionality_match and group_match:
                    accept_status = 'Accept'
                else:
                    accept_status = 'Reject'

                # Compile result
                results.append({
                    'company_name': company.company_name,
                    'accept_status': accept_status,
                    'product_match': product_match,
                    'product_reason': product_reason,
                    'functionality_match': functionality_match,
                    'functionality_reason': functionality_reason,
                    'group_match': group_match,
                    'group_reason': group_reason,
                })

            # Store results in session or pass directly to template
            request.session['search_results'] = results
            return JsonResponse({'redirect_url': '/search_results/'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def search_results(request):
    results = request.session.get('search_results', [])
    return render(request, 'search_results.html', {'results': results})

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
