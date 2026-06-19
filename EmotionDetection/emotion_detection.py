#!/usr/bin/python
# -*- coding: UTF-8 -*-

import waston_nlp
import json 

def emotion_detection(text_to_analyze):
    emotional_model = waston_nlp.load('emotion-en')

    response = emotional_model.run(text_to_analyze)
    dominant_emotion = response['emotion']['predictions'][0]['label']
    response['detected_emotion'] = dominant_emotion

    response_json = json.dumps(response, indent=4)
    response_dict = json.loads(response_json)

    return response_dict

if __name__ == "__main__":
    sample_text = "I love this new technology."
    sample_text = 'I am so happy I am doing this.'
    result = emotion_detection(sample_text)
    print("Detected Emotion:", result['detected_emotion'])