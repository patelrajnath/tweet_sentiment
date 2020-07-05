import json

template = {
        'context': "",
        'qas': [
            {
                'id': "",
                'is_impossible': False,
                'question': "",
                'answers': [
                    {
                        'text': "",
                        'answer_start': 0
                    }
                ]
            }
        ]
    }

train_data = []

import pandas as pd

train = pd.read_csv('data/test.csv')
for id, row in train.iterrows():
    # print(row['text'])
    # print(row['selected_text'])
    # print(row['sentiment'])
    # print(row['textID'])
    template['context'] = row['text']
    template['qas'][0]['id'] = row['textID']
    template['qas'][0]['question'] = row['sentiment']
    # template['qas'][0]['answers'][0]['text'] = row['selected_text']
    # try:
    #     template['qas'][0]['answers'][0]['answer_start'] = row['text'].index(row['selected_text'])
    # except AttributeError:
        # print(id, row['text'], row['selected_text'])

    train_data.append(template)
    # print(template)
    # if id == 10:
    #     exit()
    # print(train_data)
print(train_data)
with open('data/test_processed.json', 'w') as f:
    json.dump(train_data, f)
