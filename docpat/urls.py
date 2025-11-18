"""
URL configuration for docpat project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from docpat import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('diabetes/',views.diabetes),
    path('liver/',views.liver),
    path('faqs/',views.faq),
    path('breast/',views.breast),
    path('predictb/',views.predictb,name='predictb'),
    path('predictd/',views.predictd,name='predictd'),
    path('predictl/',views.predictl,name='predictl'),
    path('predicth/',views.predicth,name='predicth'),
    path('heart/',views.heart),
    path('diabetes_info/',views.diabetes_info),
    path('breast_cancer_info/',views.breast_cancer_info),
    path('heart_disease_info/',views.heart_disease_info),
    path('liver_disease_info/',views.liver_disease_info),
    path('',views.homepage),
    path('GI_diseases/', views.gi, name='GI_diseases'),
    path('predictg/', views.predictg, name='predictg'),
    path('GI_Diseases_info/',views.GI_Diseases_info),
]
