from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Create dummy data to use for training.
# train_data = [
#     {
#         'context': "This is the first context",
#         'qas': [
#             {
#                 'id': "00001",
#                 'is_impossible': False,
#                 'question': "Which context is this?",
#                 'answers': [
#                     {
#                         'text': "the first",
#                         'answer_start': 8
#                     }
#                 ]
#             }
#         ]
#     },
#     {
#         'context': "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales, "
#                    "and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
#         'qas': [
#             {
#                 'id': "00002",
#                 'is_impossible': False,
#                 'question': "What was the cost to society?",
#                 'answers': [
#                     {
#                         'text': "low cost",
#                         'answer_start': 225
#                     }
#                 ]
#             },
#             {
#                 'id': "00003",
#                 'is_impossible': False,
#                 'question': "What was the name of the 1937 treaty?",
#                 'answers': [
#                     {
#                         'text': "Bald Eagle Protection Act",
#                         'answer_start': 167
#                     }
#                 ]
#             }
#         ]
#     }
# ]

# # Save as a JSON file
# os.makedirs('data', exist_ok=True)
# with open('data/train.json', 'w') as f:
#     json.dump(train_data, f)

# with open('data/train_processed.json', 'r') as f:
#     train_data = json.load(f)
    
with open('data/test_processed.json', 'r') as f:
    test_data = json.load(f)

# train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]


train_args = {
    'use_multiprocessing': False,
    # 'use_early_stopping': True,
    # 'early_stopping_patience': 7,
    'weight_decay': 0.000001,
    'do_lower_case': False,
    "wandb_project": False,
    'learning_rate': 3e-5,
    'num_train_epochs': 10,
    'max_seq_length': 384,
    'doc_stride': 128,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    # 'train_batch_size': 20,
    # 'gradient_accumulation_steps': 8,
    'save_eval_checkpoints': False,
    'save_model_every_epoch': True
}
arch = 'distilbert'
m = 'distilbert-base-uncased-distilled-squad'
m = 'outputs/'
# m = 'outputs/checkpoint-1644-epoch-2'

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel(arch, m,
                               args=train_args,
                               use_cuda=True
                               )

# Train the model with JSON file
# model.train_model()

# model.train_model(train_data)

# The list can also be used directly
# model.train_model(train_data)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
# result, text = model.eval_model('data/train.json')

# print(result)
# print(text)

print('-------------------')

# to_predict = [{'context': "Method of Recharging a Transportation Card. ",
#                'qas': [{'question': 'Huawei PAY is not supported for transportation card recharge?', 'id': '0'}]
#                },
#               {'context': 'What can I do if I cannot see the entrance for adding Huawei Pay traffic cards? ',
#                'qas': [{'question': "I can't get a transit card?", 'id': '1'}]
#                }
#               ]

import pandas as pd

pred, prob = model.predict(test_data, n_best_size=1)
out_df = pd.DataFrame(pred)
out_df.to_csv('submission.csv')