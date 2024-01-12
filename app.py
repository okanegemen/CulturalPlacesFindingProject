from flask import Flask, request, redirect, url_for, send_from_directory, jsonify,flash
from flask_restful import Resource, Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from createFeaturesAndSearch import *
import config as cfg
from PIL import Image
import json
import requests

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

        db = pd.read_csv("/Users/okanegemen/CulturalPlacesFindingProject/metaData/dataWithImages.csv")
        label = data["foundedImage"][0]

        name = db[db["LABELS"]==label]["NAMES"].values[0]

        name = " ".join(name.split("_")[:-1])
        resp = requests.get(f"https://tr.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={name}")
        info = resp.json()
        data["info"] = info['query']
       
        return data
    
    @app.route("/postData",methods= ["POST"])
    def post():

        if 'image' not in request.files:
            return "there is no image in request file",404
        file = request.files['image']
        location = request.form["location"]

        if file.filename == '':
            
            return "there is no file",302
        
        

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            



            imgPath = "/Users/okanegemen/CulturalPlacesFindingProject/TestImages/" + filename
            file.save(imgPath) 

            

            dicti = search(modelList= cfg.MODELS,
                            queryImage = imgPath,
                            nImg=5)
            
            if len(os.listdir("/Users/okanegemen/CulturalPlacesFindingProject/TestImages"))>2:
                name = os.listdir("/Users/okanegemen/CulturalPlacesFindingProject/TestImages")[0]

                os.remove(f"/Users/okanegemen/CulturalPlacesFindingProject/TestImages/{name}")


            

            if type(dicti) is str:
                return jsonify({"title": "PLACE NOT FOUND","info":dicti}),404
            

            dumped = json.dumps(dicti,cls=NpEncoder)

            
            with open(f"/Users/okanegemen/CulturalPlacesFindingProject/Output/requests.json","w") as f:
                
                json.dump(dicti,f,cls=NpEncoder)
            f.close()
            jsonList = sorted(os.listdir("/Users/okanegemen/CulturalPlacesFindingProject/Output"))

            with open(f"/Users/okanegemen/CulturalPlacesFindingProject/Output/{jsonList[-1]}","r") as f:

                data = json.load(f)
            f.close()

            

            db = pd.read_csv("/Users/okanegemen/CulturalPlacesFindingProject/metaData/dataWithImages.csv")

            
            
            label = data["foundedImage"][0]

            name = db[db["LABELS"]==label]["NAMES"].values[0]

            name = " ".join(name.split("_")[:-1])

            if name == "Anitkabir":
                name = "Anıtkabir"

            resp = requests.get(f"https://tr.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={name}")

            info = resp.json()
            
            extracted = info["query"]
            
            extracted = extracted["pages"]
            extracted = extracted[list(extracted.keys())[0]]
            extracted = extracted["extract"]
            # print(info["query"].keys())

            willReturn = {"title" : name, "info":extracted}
            




            
        
            return jsonify(willReturn),200
        else:
            error = 'Allowed file types are png, jpg, jpeg, gif'
            return jsonify({"error": error}),400

    





    


    

    
        



# api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":

    
    port = int(os.environ.get("PORT", cfg.PORT))
    app.run(host=f'{cfg.HOST}',port = port,debug=True)


