import os
import pickle
import numpy as np
import json

model = None
encoders = None
target_mapping = None



def load_model():
    """
    Loads and returns the trained machine learning model, label encoders, and target class mapping.

    This function ensures the model, encoders, and target mapping are loaded only once and cached globally.
    It loads the following from the ../model/ directory:
        - model.pkl : Trained classification model
        - encoders.pkl : Dictionary of LabelEncoder objects for categorical features
        - target_mapping.json : Mapping of encoded target labels to original class names

    Returns:
        tuple: A tuple containing:
            - model (sklearn.base.BaseEstimator): The trained ML model
            - encoders (dict): A dictionary of fitted LabelEncoder instances
            - target_mapping (dict): A mapping of target class indices (int) to class labels (str)
    """
    global model, encoders, target_mapping

    if model is None or encoders is None or target_mapping is None:
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, '..', 'model', 'model.pkl')
        encoder_path = os.path.join(base_path, '..', 'model', 'encoders.pkl')
        mapping_path = os.path.join(base_path, '..', 'model', 'target_mapping.json')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)

        with open(mapping_path, 'r') as f:
            target_mapping = json.load(f)
            target_mapping = {int(k): v for k, v in target_mapping.items()}

    return model, encoders, target_mapping



def preprocess_input(form_data):
    """
    Preprocesses user-submitted form data into a numerical input vector for model prediction.

    This function:
    - Transforms categorical fields using pre-fitted LabelEncoders.
    - Encodes binary and multi-label fields like known languages and frameworks.
    - Computes aggregate features based on combinations of known tools/languages.

    Args:
        form_data (ImmutableMultiDict): Form data from a Flask request, typically from `request.form`.
            Expected fields:
                - interest_area (str)
                - coding_interest_level (str)
                - preferred_domain (str)
                - current_degree (str)
                - field_of_study (str)
                - current_status (str)
                - job_type_preference (str)
                - job_experience (str: 'None', '1+', or '3+')
                - known_languages (list of str)
                - frameworks_known (list of str)

    Returns:
        list: A list of integers representing the processed feature vector in the following order:
            [interest_area, coding_interest_level, preferred_domain,
             current_degree, field_of_study, current_status, job_type_preference,
             job_experience, C, SQL, JavaScript, React, Flask, 
             Python or R, Java or C++, TF/PT/Sklearn, Django or Node]
    """
    _, encoders, _ = load_model()

    interest_area = encoders['interest_area'].transform([form_data['interest_area'].lower()])[0]
    coding_interest_level = encoders['coding_interest_level'].transform([form_data['coding_interest_level'].lower()])[0]
    preferred_domain = encoders['preferred_domain'].transform([form_data['preferred_domain'].lower()])[0]
    current_degree = encoders['current_degree'].transform([form_data['current_degree'].lower()])[0]
    field_of_study = encoders['field_of_study'].transform([form_data['field_of_study'].lower()])[0]
    current_status = encoders['current_status'].transform([form_data['current_status'].lower()])[0]
    job_type_preference = encoders['job_type_preference'].transform([form_data['job_type_preference'].lower()])[0]

    experience_map = {
        'None': 0,
        '1+': 1,
        '3+': 3
        }
    job_experience = experience_map.get(form_data['job_experience'], 0)

    known_languages = ['c', 'sql', 'javascript', 'python', 'r', 'java', 'c++']
    frameworks = ['react','node','flask', 'django', 'tensorflow','pytorch','scikit-learn']

    known_languages_vector = [1 if lang in form_data.getlist('known_languages') else 0 for lang in known_languages]
    frameworks_vector = [1 if fw in form_data.getlist('frameworks_known') else 0 for fw in frameworks]

    python_r = 1 if(known_languages_vector[3] + known_languages_vector[4] >= 1) else 0
    java_cpp = 1 if(known_languages_vector[-2] + known_languages_vector[-1] >= 1) else 0
    tf_pt_sk = 1 if(frameworks_vector[-1] + frameworks_vector[-2] + frameworks_vector[-3] >= 1) else 0
    django_node = 1 if(frameworks_vector[1] + frameworks_vector[3] >= 1) else 0
    
    return [interest_area, coding_interest_level, preferred_domain,
       current_degree, field_of_study, current_status,job_type_preference, job_experience,
       known_languages_vector[0], known_languages_vector[1], known_languages_vector[2],
       frameworks_vector[0], frameworks_vector[2], python_r, java_cpp, tf_pt_sk, django_node]



def predict_job(input_data):
    """
    Predicts the most suitable job role based on the preprocessed input features.

    This function:
    - Loads the trained model and target mapping.
    - Reshapes the input data to match the model's expected input shape.
    - Performs prediction using the trained classification model.
    - Converts the predicted class index to the corresponding job label.

    Args:
        input_data (list): A list of numerical features representing a user's profile,
                           formatted as per the model’s input requirements.

    Returns:
        str: The predicted job role label (e.g., 'Data Scientist', 'Backend Developer').
    """
    model, _, target_mapping = load_model()
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return target_mapping[prediction[0]]





