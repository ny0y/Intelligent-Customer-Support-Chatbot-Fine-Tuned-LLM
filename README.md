_____________________________________________________________
The project files include 3 models:
________________________________________________________________

1- chatbot model  --- uses RFC & TF-IDF
src\chatbot\intents_classifier.py
src\chatbot_service\chatbot.py
src\app.py

2- Bert model
src\bert\bert_classifier.py
src\bert_service\bert_chatbot_service.py
mainBert.py
Other:
src\bert 
src\bert\bert_fine_tuned_model


3- Distailbert model
src\Distailbert\Distailbert_classifier.py
src\Distailbert\Distailbert_service.py
mainDi
stail.pyOther:
src\models\bert 
src\models\bert_fine_tuned_model



_________________________________________________________________
Project Setup and Running Instructions
_____________________________________________________________

Step 1: Create a New Virtual Environment
To avoid package conflicts, it's best to create a new virtual environment for this project.

Open your terminal or command prompt.

Run the following command to create a virtual environment:
python -m venv myenv


Step 2: Activate the Virtual Environment
After creating the virtual environment, you need to activate it.

On Windows:
myenv\Scripts\activate

On macOS/Linux:
source myenv/bin/activate

Step 3: Install Required Packages
Once the virtual environment is activated, install the necessary packages using pip. You can do this by running the following command:
pip install -r requirements.txt

This command will install all the dependencies listed in the requirements.txt file.

Step 4: Run the Application
To start the Flask application, execute the following command in your terminal, adjust the path to the app.py file based on your location:

python src/app.py

Step 5: Testing the API
You can test the API using an API testing application like Postman. Follow these steps:

Open Postman.

Set the request type to POST.

Enter the correct route for the API.

http://127.0.0.1:5000/chat/chatbot

http://127.0.0.1:5000/chat/bert

http://127.0.0.1:5000/chat/bert/Distailbert


Go to the "Body" tab, select "raw", and choose "JSON" from the dropdown menu.

Enter the following JSON structure in the body:


{
    "message": "Your message here"
}

Click "Send" to test the API.

e.g message:
    "message": "What is your return policy?"
    "message": "Hello"
    "message": "I need to update my address"


<<<<<<< HEAD
worked by Ahmed Almalki, Ossama Abbas
=======
>>>>>>> a9684214 (Initial commit)
