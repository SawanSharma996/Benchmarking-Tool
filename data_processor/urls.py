from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('fetch_additional_data/', views.fetch_additional_data, name='fetch_additional_data'),
    path('export_excel/', views.export_excel, name='export_excel'),
    path('generate_related_keywords/', views.generate_related_keywords, name='generate_related_keywords'),
    path('search_companies/', views.search_companies, name='search_companies'),
    path('display_excel/', views.display_excel, name='display_excel'),
    path('search_results/', views.search_results, name='search_results'),
    path('export_search_results/', views.export_search_results, name='export_search_results'),
    path('export_pdf/', views.export_pdf, name='export_pdf'),
    path('company/<int:company_id>/', views.company_detail, name='company_detail'),
     path('get_processed_count/', views.get_processed_count, name='get_processed_count'),
     path('get_past_uploads/', views.get_past_uploads, name='get_past_uploads'),

]