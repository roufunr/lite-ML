import http.server
import socketserver
import os
import shutil
import tempfile
import logging

MODELS_DIR = 'models'

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + '/logger', exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/server.log',
                    filemode='a')

logger = logging.getLogger(__name__)

class ModelRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the query parameters to get the model index
        query_components = self.path.split('?')
        if len(query_components) == 2:
            query_params = dict(qc.split('=') for qc in query_components[1].split('&'))
            model_idx = query_params.get('model_idx')
        else:
            model_idx = None

        if model_idx is not None:
            tf_model_path = os.path.join(MODELS_DIR, 'tf', f'{model_idx}')
            lite_model_path = os.path.join(MODELS_DIR, 'lite', f'{model_idx}.tflite')
            
            if os.path.exists(tf_model_path) and os.path.exists(lite_model_path):
               
                with tempfile.TemporaryDirectory() as temp_dir:
                    os.makedirs(temp_dir + "/model")
                    tf_temp_path = os.path.join(temp_dir + "/model", 'tf')
                    lite_temp_path = os.path.join(temp_dir + "/model", model_idx + ".tflite")
                    
                    shutil.copytree(tf_model_path, tf_temp_path)
                    shutil.copy2(lite_model_path, lite_temp_path)

                    zip_file_path = os.path.join(temp_dir, f'model.zip')
                    shutil.make_archive(zip_file_path[:-4], 'zip', temp_dir + "/model")

                    logger.info(f"model -> {model_idx} is serving")
                    with open(zip_file_path, 'rb') as zip_file:
                        self.send_response(200)
                        self.send_header('Content-type', 'application/zip')
                        self.end_headers()
                        self.wfile.write(zip_file.read())
                    logger.info(f"model -> {model_idx} is serving DONE")
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Model not found')
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad request')

# Set up the HTTP server
PORT = 8000
Handler = ModelRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    logger.info(f"Server running on port {PORT}")
    httpd.serve_forever()
