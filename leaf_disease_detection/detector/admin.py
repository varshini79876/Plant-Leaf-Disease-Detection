from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'plant_name', 'disease_name', 'confidence', 'is_healthy', 'created_at']
    list_filter = ['is_healthy', 'plant_name']
    search_fields = ['plant_name', 'disease_name']
    readonly_fields = ['created_at']
