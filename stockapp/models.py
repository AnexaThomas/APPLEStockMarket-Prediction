from django.db import models

# Create your models here.

class user(models.Model):
    name=models.CharField(max_length=255)
    email=models.EmailField(max_length=255)
    password=models.CharField(max_length=255)
    rpwd=models.CharField(max_length=255)

class contacts(models.Model):
    name=models.CharField(max_length=255)
    email=models.EmailField(max_length=255)
    message=models.CharField(max_length=255)
    
class feed(models.Model):
    name=models.CharField(max_length=255)
    email=models.EmailField(max_length=255)
    message=models.CharField(max_length=255)
    

class company(models.Model):
    cname = models.CharField(max_length = 255)
    cimg = models.ImageField(upload_to = 'companies')
    cdesc = models.CharField(max_length = 255)

    def __str__(self):
        return self.cname