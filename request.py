import requests

# Specify the URL of your Flask API endpoint
api_url = 'http://localhost:5000/upload'  # Update the URL if your API is running on a different port or host

# Specify the image file you want to upload
image_file = 'C:/Users/bpran/Desktop/Mini project/Project/Dataset/Wounds - Degree/Degree 1/3-s2.0-B9780128000342002238-f0300223-02-9780128000342.jpg'  # Replace 'path_to_your_image.jpg' with the actual image file path

# Create a dictionary to hold the file data
files = {'file': open(image_file, 'rb')}

# Make a POST request to the API endpoint
response = requests.post(api_url, files=files)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Print the predicted class returned by the API
    print('Predicted class:', response.json()['predicted_class'])
else:
    # Print an error message if the request was not successful
    print('Error:', response.json()['error'])
