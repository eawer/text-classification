import re
import psycopg2
import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from transformers import AutoTokenizer
import tritonclient.http as httpclient


# List of labels, where the position of each label corresponds to it's integer representation
LABELS = [
    'J80', 'E039', 'I4891', 'F319', 'E119', 'F0280', 'I609', 'R65.21',
    'K7030', 'K766', 'M810', 'B182', 'D696', 'I469', 'N186', 'J449',
    'I2699', 'Z79.4', 'F10239', 'I25.2', 'E6601', 'A419', 'I714', 'R570',
    'J15211', 'I472', 'I10', 'E780', 'F329', 'E46', 'K219', 'I6529', 'I619',
    'I6350', 'J9620', 'G936', 'M069', 'N189', 'I739', 'N179', 'I200',
    'I214', 'I509', 'C7931', 'I2510', 'I129', 'J690', 'R569', 'C787',
    'F341'
]
# List of words that should be removed from the text
STOPWORDS = ['with', 'and', 'of', 'to', 'the', 'a', 'in', 'on', 'for', 'old']

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", return_tensors='pt')
triton_client = httpclient.InferenceServerClient(url='inference:8000')
db_conn = psycopg2.connect(dbname='test', user='user', password='password', host='db')

def clean_text(text: str) -> str:
    """Clean the text and remove the stopwords

    Parameters
    ----------
    text : str
        Text to clean

    Returns
    -------
    str
        Clean text
    """
    result = text.lower()
    result = re.sub(r'\[\*{2}[^]]+\*{2}\]', ' ', result)
    result = re.sub(r'[-,.?()#;><]', ' ', result)
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(fr'\\b({"|".join(STOPWORDS)})\\b', ' ', result)
    result = re.sub(r'\s{2,}', ' ', result)
    result = result.rstrip()

    return result

def preprocess(text:str, tokenizer:AutoTokenizer) -> dict:
    """Clean and tokenize the text

    Parameters
    ----------
    text : str
        Text to tokenize
    tokenizer : AutoTokenizer
        Approptiate tokenizer from the huggingface

    Returns
    -------
    dict
        Dictionary with "input_ids", "attention_mask" and "token_type_ids" keys
    """
    return tokenizer(clean_text(text), padding='max_length', truncation=True, max_length=128, return_tensors="np")

def get_prediction(text:str):
    """Send the text to a triton inference server and get the prediction

    Parameters
    ----------
    text : str
        Text to classify

    Returns
    -------
    _type_
        Text category
    """
    tokenized_data = preprocess(text, tokenizer)

    inputs = [0, 0, 0]
    inputs[0] = httpclient.InferInput('input_ids', [1, 128], 'INT64')
    inputs[1] = httpclient.InferInput('attention_mask', [1, 128], 'INT64')
    inputs[2] = httpclient.InferInput('token_type_ids', [1, 128], 'INT64')

    # Initialize the data
    inputs[0].set_data_from_numpy(tokenized_data['input_ids'], binary_data=False)
    inputs[1].set_data_from_numpy(tokenized_data['attention_mask'], binary_data=False)
    inputs[2].set_data_from_numpy(tokenized_data['token_type_ids'], binary_data=False)

    output = httpclient.InferRequestedOutput('logits',  binary_data=False)
    logits = triton_client.infer('maverick', model_version='1', inputs=inputs, outputs=[output])
    scores = logits.as_numpy('logits')
    predictions = np.argmax(scores, 1)

    return LABELS[predictions[0]]

def check_user(user_id:int) -> bool:
    """Checks whether the user is a new one
    Parameters
    ----------
    user_id : int
        Id to check

    Returns
    -------
    bool
        True if user is new, False otherwise
    """
    with db_conn.cursor() as cursor:
        cursor.execute(f"INSERT INTO users(user_id) VALUES({user_id}) ON CONFLICT DO NOTHING RETURNING user_id;")
        return cursor.fetchone() is not None

@app.post("/predict")
async def predict(request: Request) -> tuple[int, str, str, bool]:
    """Function that handles incoming post request for the "/predict" endpoint

    Parameters
    ----------
    request : Request
        Request object containing request parameters as json

    Returns
    -------
    tuple[int, str, str, bool]
        int - user id
        str - text
        str - text category
        bool - whether user is a returning one
    """
    json = await request.json()

    assert 'user_id' in json, '"user_id" property is missing'
    assert 'text' in json, '"text" property is missing'

    text = json['text']
    user_id = json['user_id']

    return user_id, text, get_prediction(text), check_user(user_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=5000, workers=2)

