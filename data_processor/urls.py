from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('fetch_additional_data/', views.fetch_additional_data, name='fetch_additional_data'),
    path('export_excel/', views.export_excel, name='export_excel'),
    path('generate_related_keywords/', views.generate_related_keywords, name='generate_related_keywords'),
    path('search_companies/', views.search_companies, name='search_companies'),
    path('display_excel/', views.display_excel, name='display_excel'),
    path('search_companies/', views.search_companies, name='search_companies'),
    path('search_results/', views.search_results, name='search_results'),

]