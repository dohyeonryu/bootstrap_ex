from django.shortcuts import render
from bootapp.skkuu import SKKU_SENTIMENT

# Create your views here.

def home(request):
	skku = SKKU_SENTIMENT()
	return render(request, 'home.html', {
		#'skku' : skku("좋아요"),
		'skku_device' : skku.device,
		'skku_epochs' : skku.args.epochs	
	})


def about(request):
    return render(request, 'about.html')