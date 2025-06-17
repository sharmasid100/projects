import http.client
import json


cities = {
    "NewYork": "us",
    "Chicago": "us",
    "SanFrancisco": "us",
    "London": "gb",
    "Sydney": "au",
    "Singapore": "sg",
    "Berlin": "de",
    "Paris": "fr",
    "Dubai": "ae"
}


def return_links(role , location):
    conn = http.client.HTTPSConnection("jsearch.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "44983e66e3mshb549a3af95b11d6p17e4a8jsn68251e81e59f",
        'x-rapidapi-host': "jsearch.p.rapidapi.com"
    }

    if location in cities:
        country = cities[location]
    else:
        country = "in"

    req_link = f"/search?query={role}%20jobs%20in%20{location}&page=1&num_pages=2&country={country}&date_posted=all"
    conn.request("GET", req_link , headers=headers)

    res = conn.getresponse()
    data = res.read()

    return json.loads(data.decode("utf-8"))




def parse_job_data(api_response_json):
    job_data = []

    for job in api_response_json.get("data", []):
        job_entry = {
            "title": job.get("job_title"),
            "company": job.get("employer_name"),
            "location": job.get("job_location"),
            "description": job.get("job_description", "")[:250] + "...",  # brief
            "link": job.get("job_apply_link"),
            "logo": job.get("employer_logo"),
            "posted": job.get("job_posted_at"),
            "type": job.get("job_employment_type"),
        }
        job_data.append(job_entry)

    return job_data
