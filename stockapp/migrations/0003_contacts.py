# Generated by Django 4.0.4 on 2022-10-28 10:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockapp', '0002_company'),
    ]

    operations = [
        migrations.CreateModel(
            name='contacts',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('email', models.EmailField(max_length=255)),
                ('message', models.CharField(max_length=255)),
            ],
        ),
    ]
