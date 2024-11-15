# Generated by Django 5.0.4 on 2024-10-23 11:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CompanyInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('company_name', models.CharField(max_length=255)),
                ('website_address', models.URLField(max_length=255)),
                ('trade_description', models.TextField()),
                ('full_overview', models.TextField()),
                ('business_description', models.TextField(blank=True, null=True)),
                ('products_services', models.TextField(blank=True, null=True)),
                ('part_of_group', models.TextField(blank=True, default=False)),
                ('functionality', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
