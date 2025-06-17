from flask import Flask, render_template, request
from utils import preprocess_input, predict_job
from api_handler import return_links, parse_job_data

app = Flask(__name__)

attributes = {
    "interest_area": ["Creativity", "Logic", "Both"],
    "coding_interest_level": ["Beginner", "Intermediate", "Advanced"],
    "preferred_domain": ["Frontend", "Backend", "Full Stack", "AI/ML", "Data Science", "Cybersecurity", "DevOps"],
    "current_degree": ["Bachelors", "Masters", "Diploma"],
    "field_of_study": ["CS", "Non-CS"],
    "current_status": ["Pursuing Degree", "Intern", "Trainee", "Job Seeker", "Working Professional"],
    "known_languages": ["Python", "JavaScript", "Java", "C++", "C", "R", "SQL"],
    "frameworks_known": ["React", "Node", "Flask", "Django", "TensorFlow", "PyTorch", "Scikit-learn"],
    "job_type_preference": ["Internship", "Full-time Job", "Freelance", "Remote", "Onsite"],
    "job_experience": ["None", "1+", "3+"],
}

# Landing page
@app.route('/')
def landing():
    return render_template("landing.html")

# Input form page
@app.route('/form')
def form():
    return render_template("index.html", attributes=attributes)

#Learn more page
@app.route('/learn-more')
def learn_more():
    return render_template("learn_more.html")

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form
        input_vector = preprocess_input(form_data)
        prediction = predict_job(input_vector).upper()


        location = form_data['location'].replace(" ", "")
        api_data = return_links(prediction, location)
        jobs = parse_job_data(api_data)

        return render_template("result.html", prediction=prediction, jobs=jobs)

if __name__ == '__main__':
    app.run(debug=True)
