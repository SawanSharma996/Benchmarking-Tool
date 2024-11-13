from django.db import models
import uuid
from django.contrib.auth.models import User


class CompanyInfo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE , null=True, blank=True)    
    company_name = models.CharField(max_length=255, blank= False, null= False)
    website_address = models.URLField(max_length=255)
    trade_description = models.TextField()
    full_overview = models.TextField()
    business_description = models.TextField(blank=True, null=True)
    products_services = models.TextField(blank=True, null=True)
    part_of_group = models.TextField(blank=True,default=False)
    products_servicesD = models.TextField(null=True, blank=True)
    batch_id = models.UUIDField(default=uuid.uuid4, editable=False)
    screenshot = models.ImageField(upload_to='screenshots/', null=True, blank=True)


    def __str__(self): 
        return self.company_name
