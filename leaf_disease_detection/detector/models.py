from django.db import models
from django.utils import timezone


class PredictionHistory(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_class = models.CharField(max_length=200)
    confidence = models.FloatField()
    plant_name = models.CharField(max_length=100)
    disease_name = models.CharField(max_length=100)
    is_healthy = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.plant_name} — {self.disease_name} ({self.confidence:.1f}%)"
