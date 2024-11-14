from flask import Flask, request, jsonify
from inference import ModelInference
import pandas as pd

app = Flask(__name__)
model_inference = ModelInference()

@app.route('/get_questions', methods=['POST'])
def get_questions():
    data = request.json
    prompt = data.get("prompt", "")
    
    metadata = model_inference.predict(prompt)

    # Load question data for search
    df = pd.read_json(config.DATA_PATH)
    filtered_df = df[(df['subject'] == metadata['subject']) &
                     (df['chapter'] == metadata['chapter']) &
                     (df['topic'] == metadata['topic']) &
                     (df['difficulty'] == metadata['difficulty'])]

    results = filtered_df.head(10).to_dict(orient="records")
    return jsonify({"questions": results})

if __name__ == '__main__':
    app.run(debug=True)
