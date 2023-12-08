This code is designed to be deployed on a GCP instance, with a minimum configuration of:
Machine type: e2-standard-8
CPU platform: Intel Broadwell
Architecture: x86/64

Once the instance is deployed, create a virtual environment, installs requirements.txt, and assigns the following roles to your service account: Vertex AI User, AI Platform Developer, and a custom role with the following permissions:
aiplatform.endpoints.predict
storage.objects.create
storage.objects.delete
storage.objects.get

After completing the above steps, create two folders using the following commands:
bash
mkdir audios
mkdir artifacts

Keep in mind that my username (german) is hardcoded in the code; therefore, /home/german/main.py is the project directory. Adjust it according to your situation.

Launch the application in SSH with the following command to avoid errors and prevent the Streamlit of Trulens from overlapping:

bash
streamlit run main.py --server.fileWatcherType none --server.port 8502
When you press the Trulens button, find it in the browser at the same public IP address of your instance on port 8501.
