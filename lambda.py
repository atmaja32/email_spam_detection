import boto3
import json
import email
import os

ENDPOINT = os.environ['SageMakerEndPoint']

############### UTILS ###############
import string
import sys
import numpy as np

from hashlib import md5

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_seq(sequences, vocab_len):
    res = np.zeros((len(sequences), vocab_len))
    for i, sequence in enumerate(sequences):
       res[i, sequence] = 1. 
    return res

def one_hot_encoding(messages, vocab_len):
    encoded = []
    for message in messages:
        temp = one_hot(message, vocab_len)
        encoded.append(temp)
    return encoded

def text_to_word(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((character, split) for character in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    sequence = text.split(split)
    return [i for i in sequence if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
   
    return hashing(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
   
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda x: int(md5(x.encode()).hexdigest(), 16)

    seq = text_to_word(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(x) % (n - 1) + 1) for x in seq]

############### END ###############

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    s3 = boto3.client('s3')
    messageRaw = s3.get_object(Bucket=bucket, Key=key)

    email_object = email.message_from_bytes(messageRaw['Body'].read())
    from_email = email_object.get('From')
    email_body = email_object.get_payload()[0].get_payload()
    
    print(from_email)
    print(email_body)

    sage_end = ENDPOINT
    print(ENDPOINT)
    sage_runtime = boto3.client('runtime.sagemaker')

    
    vocab_len = 9013
    input_mail = [email_body.strip()]
    one_hot = one_hot_encoding(input_mail, vocab_len)

    process_mail = vectorize_seq(one_hot, vocab_len)
    json_data = json.dumps(process_mail.tolist())

    #checking for spam in mail
    modelResponse = sage_runtime.invoke_endpoint(EndpointName=sage_end, ContentType='application/json', Body=json_data)
    mail_result = json.loads(modelResponse["Body"].read())

    if mail_result['predicted_label'][0][0] == 0:
        label = 'Ok'
    else:
        label = 'Spam'
    
    spam_prediction = round(mail_result['predicted_probability'][0][0], 4)
    spam_prediction = spam_prediction*100

    print("Spam: ",label)
    print("Prediction: ", spam_prediction)

    recipient_mail = email_object.get('To')
    message = "We received your email sent at " + str(recipient_mail) + " with the subject " + str(email_object.get('Subject')) + ".\nHere is a 240 character sample of the email body:\n\n" + email_body[:240] + "\nThe email was categorized as " + str(label) + " with a " + str(spam_prediction) + "% confidence."
    
    email_client = boto3.client('ses')
    reply_email = email_client.send_email(
        Destination={'ToAddresses': [from_email]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam analysis of your email',
            },
        },
        Source=str(recipient_mail),
    )

    return {}