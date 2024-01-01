from flask import Flask, request, redirect, url_for, send_from_directory, jsonify,flash
from flask_restful import Resource, Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from createFeaturesAndSearch import *
import config as cfg
from PIL import Image
import json

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
api = Api(app)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)






class GetPredictionOutput(Resource):

    def __init__(self):
        self.dumped = None
        

    @app.route("/getPreds",methods= ["GET"])
    def get():

        jsonList = sorted(os.listdir("/Users/okanegemen/CulturalPlacesFindingProject/Output"))

        with open(f"/Users/okanegemen/CulturalPlacesFindingProject/Output/{jsonList[-1]}","r") as f:

            data = json.load(f)
        f.close()
        return data
    
    @app.route("/postData",methods= ["POST"])
    def post():

        if 'image' not in request.files:
            return "there is no image in request file",404
        file = request.files['image']

        if file.filename == '':
            
            return "there is no file",302
        
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)

            imgPath = "/Users/okanegemen/CulturalPlacesFindingProject/TestImages/" + filename
            file.save(imgPath) 

            img = cv.imread(imgPath)
            transformed = cfg.TRANSFORMS(img)
            transformed = torch.unsqueeze(transformed,dim=0)
            dicti = search(modelList= cfg.MODELS,
                            queryImage = transformed,
                            nImg=5)

            
            
            dumped = json.dumps(dicti,cls=NpEncoder)

            
            with open(f"/Users/okanegemen/CulturalPlacesFindingProject/Output/requests.json","w") as f:
                json.dump(dicti,f,cls=NpEncoder)
            f.close()
        
            return dumped
        else:
            error = 'Allowed file types are png, jpg, jpeg, gif'
            return jsonify({"error": error}),400


# api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":

    
    port = int(os.environ.get("PORT", cfg.PORT))
    app.run(host=f'{cfg.HOST}',port = port,debug=True)


