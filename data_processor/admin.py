from django.contrib import admin
from .models import CompanyInfo

@admin.register(CompanyInfo)
class CompanyInfoAdmin(admin.ModelAdmin):
    list_display = ('company_name', 'website_address', 'trade_description', 'full_overview', 'business_description', 'products_services','functionality', 'part_of_group')
