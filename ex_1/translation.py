from google.cloud import translate_v2
import pandas as pd
import os
import glob
import json


def create_json(file_name):  # create a json file with particular structure
    a_dict = {"sentences": []}
    with open(file_name + ".json", "w", encoding='utf-8') as f:
        f.write(json.dumps(a_dict, ensure_ascii=False, indent=4))
        f.close()


def translate_files(file_name, path, target):
    dict = {}
    translated_client = translate_v2.Client()
    f = open(path)
    data = json.load(f)
    for i in data['sentences']:
        output = translated_client.translate(i['text'], target_language=target)
        re_output = translated_client.translate(output['translatedText'], target_language=i['lang'])
        text = re_output['translatedText']
        dict['id'] = i['id']
        dict['lang'] = i['lang']
        dict['ilang'] = target
        dict['text'] = text
        f = open('/Users/wenxu/PycharmProjects/ComputationalHumanities/'+file_name+".json")
        data = json.load(f)
        data['sentences'].append(dict)

        with open(file_name + ".json", "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = r'/Users/wenxu/PycharmProjects/ComputationalHumanities/computationalhumanities-ba9e3b76e554.json'
    create_json("ja_hi")
    # path: source file
    translate_files("ja_hi", '/unzipped/dataset-ja.json',
                    target='hi')
