import roboflow
import requests  # Import the requests module

# Ensure the API key is correct and has the necessary permissions
rf = roboflow.Roboflow(api_key="YOUR_CORRECT_API_KEY")
workspace = rf.workspace("your_workspace_name")  # Replace with your actual workspace name
project = workspace.project("urine-test-strips-main")
model = project.version("14").model

try:
    prediction = model.download()
    print("Model downloaded successfully.")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 403:
        print("Failed to download model: 403 Forbidden. Please check your API key and permissions.")
    else:
        print(f"Failed to download model: {e}")