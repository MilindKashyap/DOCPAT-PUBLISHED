from django.shortcuts import render
import pickle
import os
from django.core.files.storage import FileSystemStorage
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from django.conf import settings
from xc import predict_image  # Import the function from c.py


# Path to your single image

def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The model file at {file_path} does not exist.")
    
    # Fix for scikit-learn version compatibility with monotonic_cst attribute
    # This handles the error: 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'
    # The issue occurs when models trained with older scikit-learn are loaded with newer versions
    
    # Custom unpickler to handle the monotonic_cst compatibility issue
    class CompatibleUnpickler(pickle.Unpickler):
        def load(self):
            obj = super().load()
            # After loading, fix any monotonic_cst issues
            if hasattr(obj, '__dict__'):
                # Remove problematic monotonic_cst from __dict__ if it exists
                if 'monotonic_cst' in obj.__dict__:
                    try:
                        # Try to access it - if it works, keep it
                        _ = getattr(obj, 'monotonic_cst')
                    except AttributeError:
                        # If accessing fails, remove it from __dict__
                        obj.__dict__.pop('monotonic_cst', None)
            
            # For tree-based classifiers, ensure monotonic_cst exists
            model_type = type(obj).__name__
            if 'Tree' in model_type or 'Classifier' in model_type:
                if not hasattr(obj, 'monotonic_cst'):
                    try:
                        # Set it to None for compatibility
                        setattr(obj, 'monotonic_cst', None)
                    except:
                        # If setattr fails, try adding to __dict__
                        if hasattr(obj, '__dict__'):
                            obj.__dict__['monotonic_cst'] = None
            return obj
    
    try:
        # Try loading with the custom unpickler first
        with open(file_path, 'rb') as file:
            unpickler = CompatibleUnpickler(file)
            model = unpickler.load()
        
        # Additional post-processing
        if hasattr(model, '__dict__'):
            # Clean up any problematic attributes
            if 'monotonic_cst' in model.__dict__ and not hasattr(model, 'monotonic_cst'):
                model.__dict__.pop('monotonic_cst', None)
        
        return model
    except (AttributeError, TypeError, KeyError) as e:
        error_str = str(e)
        if 'monotonic_cst' in error_str:
            # Fallback: try regular pickle load and fix afterwards
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    with open(file_path, 'rb') as file:
                        model = pickle.load(file)
                    
                    # Force fix monotonic_cst
                    if hasattr(model, '__dict__'):
                        model.__dict__.pop('monotonic_cst', None)
                    
                    # Set monotonic_cst to None for compatibility
                    model_type = type(model).__name__
                    if 'Tree' in model_type or 'Classifier' in model_type:
                        try:
                            setattr(model, 'monotonic_cst', None)
                        except:
                            if hasattr(model, '__dict__'):
                                model.__dict__['monotonic_cst'] = None
                    
                    return model
                except Exception as e2:
                    # If all else fails, raise the original error
                    raise e
        else:
            raise
def homepage(request):
        return render(request,"main.html")
def breast(request):
     return render(request,"breast.html")
def faq(request):
        return render(request,"faqs.html")

def heart(request):
        return render(request,"heart.html")

    
def liver(request):
        return render(request,"liver.html")

    
def diabetes(request):
        return render(request,"diabetes.html")
def predictb(request):
    if request.method == 'POST':
        try:
            model_path = os.path.join(settings.BASE_DIR, 'breast_cancer_svm_model.pkl')
            breast_model = load_model(model_path)

            texture_mean = float(request.POST.get('texture_mean'))
            smoothness_mean = float(request.POST.get('smoothness_mean'))
            compactness_mean = float(request.POST.get('compactness_mean'))
            concave_points_mean = float(request.POST.get('concave_points_mean'))
            symmetry_mean = float(request.POST.get('symmetry_mean'))
            fractal_dimension_mean = float(request.POST.get('fractal_dimension_mean'))
            texture_se = float(request.POST.get('texture_se'))
            area_se = float(request.POST.get('area_se'))
            smoothness_se = float(request.POST.get('smoothness_se'))
            compactness_se = float(request.POST.get('compactness_se'))
            concavity_se = float(request.POST.get('concavity_se'))
            concave_points_se = float(request.POST.get('concave_points_se'))
            symmetry_se = float(request.POST.get('symmetry_se'))
            fractal_dimension_se =float(request.POST.get('fractal_dimension_se'))
            texture_worst = float(request.POST.get('texture_worst'))
            area_worst = float(request.POST.get('area_worst'))
            smoothness_worst = float(request.POST.get('smoothness_worst'))
            compactness_worst = float(request.POST.get('compactness_worst'))
            concavity_worst = float(request.POST.get('concavity_worst'))
            concave_points_worst = float(request.POST.get('concave_points_worst'))
            symmetry_worst = float(request.POST.get('symmetry_worst'))
            fractal_dimension_worst = float(request.POST.get('fractal_dimension_worst'))
            data = [ texture_mean, smoothness_mean, compactness_mean,
                     concave_points_mean, symmetry_mean, fractal_dimension_mean,  texture_se,
                     area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                    fractal_dimension_se,  texture_worst,  area_worst, smoothness_worst,
                    compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
            prediction_result = breast_model.predict([data])
            context = {
                'prediction_result': prediction_result[0]  
            }
            return render(request, 'predict.html', context)

        except Exception as e:
            return render(request, 'error.html', {'error': str(e)}) 
        
def predictd(request):
    if request.method == 'POST':
        try:
            model_path = os.path.join(settings.BASE_DIR, 'diabetes_model.pkl')
            diabetes_model = load_model(model_path)

            Pregnancies = float(request.POST.get('Pregnancies'))
            Glucose =float(request.POST.get('Glucose'))
            BloodPressure = float(request.POST.get('BloodPressure'))
            SkinThickness = float(request.POST.get('SkinThickness'))
            Insulin = float(request.POST.get('Insulin'))
            BMI = float(request.POST.get('BMI'))
            DiabetesPedigreeFunction = float(request.POST.get('DiabetesPedigreeFunction'))
            Age = float(request.POST.get('Age'))
            inputs = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                   BMI, DiabetesPedigreeFunction, Age 
                   ]]
            prediction_result = diabetes_model.predict(inputs)


            context = {
                'prediction_result': prediction_result[0]  
            }
            return render(request, 'predict.html', context)

        except Exception as e:
            return render(request, 'error.html', {'error': str(e)}) 

def predictl(request):
    if request.method == 'POST':
        try:
            model_path = os.path.join(settings.BASE_DIR, 'liver_prediction.pkl')
            liver_model = load_model(model_path)
            age = float(request.POST.get('Age'))
            gender = float(request.POST.get('Gender'))
            total_bilirubin = float(request.POST.get('Total_Bilirubin'))
            direct_bilirubin = float(request.POST.get('Direct_Bilirubin'))
            alkaline_phosphotase = float(request.POST.get('Alkaline_Phosphotase'))
            alamine_aminotransferase = float(request.POST.get('Alamine_Aminotransferase'))
            aspartate_aminotransferase = float(request.POST.get('Aspartate_Aminotransferase'))
            total_proteins = float(request.POST.get('Total_Protiens'))
            albumin = float(request.POST.get('Albumin'))
            albumin_and_globulin_ratio = float(request.POST.get('Albumin_and_Globulin_Ratio'))

            inputs = [[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                   alamine_aminotransferase, aspartate_aminotransferase, total_proteins, 
                   albumin, albumin_and_globulin_ratio]]
        
            
            prediction_result =liver_model.predict(inputs)


            context = {
                'prediction_result': prediction_result[0]  
            }
            return render(request, 'predict.html', context)

        except Exception as e:
            return render(request, 'error.html', {'error': str(e)}) 
def predicth(request):
    if request.method == 'POST':
        try:
            model_path = os.path.join(settings.BASE_DIR, 'heart_disease_model.pkl')
            heart_model =(load_model(model_path))
            age = float(request.POST.get('age'))
            sex = float(request.POST.get('sex'))
            cp = float(request.POST.get('cp'))
            trestbps = float(request.POST.get('trestbps'))
            chol = float(request.POST.get('chol'))
            fbs = float(request.POST.get('fbs'))
            restecg = float(request.POST.get('restecg'))
            thalach = float(request.POST.get('thalach'))
            exang = float(request.POST.get('exang'))
            oldpeak = float(request.POST.get('oldpeak'))
            slope = float(request.POST.get('slope'))
            ca = float(request.POST.get('ca'))
            thal = float(request.POST.get('thal'))
            inputs = [[age, sex,cp,trestbps, chol, fbs, restecg, 
                   thalach, exang, oldpeak, 
                   slope, ca,thal]]
            
            prediction_result = heart_model.predict(inputs)


            context = {
                'prediction_result': prediction_result[0] 
            }
            return render(request, 'predict.html', context)

        except Exception as e:
            return render(request, 'error.html', {'error': str(e)}) 
def gi(request):
    return render(request, 'GI_diseases.html')
# View to handle file upload and prediction
# views.py


def predictg(request):
    if request.method == 'POST' and request.FILES.get('uploadedImage'):
        # Handle the uploaded file
        uploaded_file = request.FILES['uploadedImage']
        file_name = uploaded_file.name

        # Save the uploaded file temporarily
        fs = FileSystemStorage()
        file_path = fs.save(file_name, uploaded_file)
        full_file_path = fs.path(file_path)

        # Debugging: Ensure the file path is correct
        print(f"File uploaded to: {full_file_path}")

        # Call the predict_image function from c.py to process the image
        predicted_class = predict_image(full_file_path)

        if predicted_class is not None:
            print(f"Predicted class: {predicted_class}")  # Debugging: print the predicted class
            return render(request, 'p.html', {
                'predicted_class': predicted_class  # Only send predicted class
            })
        else:
            print("Prediction failed")  # Debugging: print if prediction failed
            return render(request, 'error.html', {'error': 'Failed to process the image or make predictions.'})

    else:
        return render(request, 'p.html') 
def GI_Diseases_info(request):
    return render(request,"gi.html")
def diabetes_info(request):
    return render(request,"diabetes_info.html")
def breast_cancer_info(request):
    return render(request,'breast_cancer_info.html')
def heart_disease_info(request):
    return render(request,'heart_disease_info.html')
def liver_disease_info(request):
    return render(request,'liver_disease_info.html')