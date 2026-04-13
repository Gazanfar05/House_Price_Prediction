from flask import Flask, render_template, jsonify, request
import json
import base64
from pathlib import Path
import traceback

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model-info')
def get_model_info():
    try:
        with open('model_info.json', 'r') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'model_info.json not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images')
def get_images():
    try:
        images = {}
        image_files = ['model_evaluation.png', 'coefficients.png', 'rmse_analysis.png']
        
        for img_file in image_files:
            if Path(img_file).exists():
                with open(img_file, 'rb') as f:
                    images[img_file] = base64.b64encode(f.read()).decode()
        
        return jsonify(images)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        import joblib
        
        with open('house_price_model.pkl', 'rb') as f:
            model = joblib.load(f)
        
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        features = model_info['features']
        feature_values = [data.get(f, 0) for f in features]
        
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        # This is a simplified prediction - in production you'd use the saved scaler
        
        prediction = model.predict([feature_values])[0]
        
        return jsonify({
            'prediction': float(prediction),
            'formatted': f"${prediction:,.2f}"
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\nStarting Dashboard Server...")
    print("Visit: http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)