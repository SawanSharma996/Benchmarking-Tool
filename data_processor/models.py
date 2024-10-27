from django.db import models
import uuid

class CompanyInfo(models.Model):
    company_name = models.CharField(max_length=255, blank= False, null= False)
    website_address = models.URLField(max_length=255)
    trade_description = models.TextField()
    full_overview = models.TextField()
    business_description = models.TextField(blank=True, null=True)
    products_services = models.TextField(blank=True, null=True)
    part_of_group = models.TextField(blank=True,default=False)
    functionality = models.TextField(null=True, blank=True)
    batch_id = models.UUIDField(default=uuid.uuid4, editable=False)

    def __str__(self):
        return self.company_name
