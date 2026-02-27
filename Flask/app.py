import os
import numpy as np
from joblib import load
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load saved artefacts ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = load(os.path.join(BASE_DIR, 'transform.save'))
model  = load(os.path.join(BASE_DIR, 'floods.save'))

# ── Feature meta-data (mirrors notebook: dataset.iloc[:,2:7]) ─────────────────
FEATURES = [
    {'key': 'cloud_cover', 'label': 'Cloud Cover',        'unit': '%',   'min': 0,    'max': 100,  'step': 1},
    {'key': 'annual',      'label': 'Annual Rainfall',     'unit': 'mm',  'min': 0,    'max': 8000, 'step': 0.1},
    {'key': 'jan_feb',     'label': 'Jan–Feb Rainfall',    'unit': 'mm',  'min': 0,    'max': 2000, 'step': 0.1},
    {'key': 'mar_may',     'label': 'Mar–May Rainfall',    'unit': 'mm',  'min': 0,    'max': 2000, 'step': 0.1},
    {'key': 'jun_sep',     'label': 'Jun–Sep Rainfall',    'unit': 'mm',  'min': 0,    'max': 5000, 'step': 0.1},
]


@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept JSON or form-data and return flood prediction."""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        feature_values = [float(data[f['key']]) for f in FEATURES]
        features_array = np.array([feature_values])

        scaled = scaler.transform(features_array)
        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]

        return jsonify({
            'prediction': prediction,
            'label':      'Flood Risk' if prediction == 1 else 'No Flood Risk',
            'confidence': round(float(proba[prediction]) * 100, 2),
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)