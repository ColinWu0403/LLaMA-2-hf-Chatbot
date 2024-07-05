from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time
from run_model import generate_response, generate_response_from_context, find_best_response, model, tokenizer, embed_model, index, responses

@csrf_exempt
def chat_view(request):
    # Your logic for handling chat interactions
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            mode = data.get('mode', 'Normal')
            
            # Process the message with the LLM model
            
            if mode == 'RAG':
                best_context = find_best_response(message, embed_model, index, responses)
                response_text = generate_response_from_context(model, tokenizer, message, best_context)

                # response = {'message': message + " (RAG)"}
            else:
                response_text = generate_response(model, tokenizer, message)
                
                # response = {'message': message + " (Normal)"}

            response = {'message': response_text}
            # time.sleep(4)
            # response = {'message': message}
            return JsonResponse(response)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        # If it's not a POST request, return some initial data (if needed)
        return render(request, 'index.html')

