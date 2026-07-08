import os
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

from .models import PredictionHistory
from .ml_utils import predict_disease, DISEASE_INFO


def index(request):
    recent = PredictionHistory.objects.all()[:6]
    return render(request, 'detector/index.html', {'recent_predictions': recent})


def predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)

    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)

    image_file = request.FILES['image']
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if image_file.content_type not in allowed_types:
        return JsonResponse(
            {'error': 'Invalid file type. Please upload JPEG, PNG, or WebP images.'},
            status=400,
        )

    file_path = default_storage.save(
        f'uploads/{image_file.name}', ContentFile(image_file.read())
    )
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)

    try:
        result = predict_disease(full_path)
        record = PredictionHistory.objects.create(
            image=file_path,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            plant_name=result['plant_name'],
            disease_name=result['disease_name'],
            is_healthy=result['is_healthy'],
        )
        return JsonResponse({
            'success': True,
            'prediction_id': record.id,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'plant_name': result['plant_name'],
            'disease_name': result['disease_name'],
            'is_healthy': result['is_healthy'],
            'disease_info': result['disease_info'],
            'top_predictions': result['top_predictions'],
            'image_url': record.image.url,
            'model_loaded': result['model_loaded'],
        })
    except Exception as e:
        return JsonResponse({'error': f'Prediction failed: {str(e)}'}, status=500)


def history(request):
    predictions = PredictionHistory.objects.all()
    return render(request, 'detector/history.html', {'predictions': predictions})


def prediction_detail(request, pk):
    prediction = get_object_or_404(PredictionHistory, pk=pk)
    disease_info = DISEASE_INFO.get(prediction.predicted_class, {})
    return render(request, 'detector/detail.html', {
        'prediction': prediction,
        'disease_info': disease_info,
    })


def about(request):
    from .ml_utils import CLASS_NAMES
    plants = sorted(set(c.split('___')[0].replace('_', ' ') for c in CLASS_NAMES))
    return render(request, 'detector/about.html', {
        'total_classes': len(CLASS_NAMES),
        'plants': plants,
        'total_plants': len(plants),
    })

