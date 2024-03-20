import requests

# Specify the API endpoint URL
url = 'http://localhost:5000/upload'  # Change the URL if your Flask app is running on a different address/port

# Path to the image file you want to upload
file_path = 'C:/Users/bpran/Desktop/Mini project/test_image.jfif'  # Replace this with the actual path of the image file

# Create a dictionary containing the file data
files = {'file': open(file_path, 'rb')}

# Send a POST request to the API endpoint with the file data
response = requests.post(url, files=files)

# Print the response content
print(response.json())
