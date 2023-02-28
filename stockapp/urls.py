from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('about', views.about, name = 'about'),
    path('contact', views.contact, name = 'contact'),
    path('blog', views.blog, name = 'blog'),
    path('single', views.single, name = 'single'),
    path('company', views.comp, name = 'comp'),
    path('help', views.help, name = 'help'),
    path('service', views.service, name = 'service'),
    path('register', views.register, name = 'register'),
    path('login/', views.login, name = 'login'),
    path('logout', views.logout, name = 'logout'),
    path('prediction', views.prediction, name = 'prediction'),
    path('predi/', views.predi, name = 'predi'),
    path('feedback/', views.feedback, name = 'feedback'),
    path('viewfeedback/', views.viewfeedback, name='viewfeedback'),
    path('forgot-password', views.forgotPassword, name = 'forgot-password'),
    path('update-password', views.updatePassword, name = 'update-password')
    
]