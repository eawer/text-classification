### How to run
1. Download the archive with the model and extract its contents into `model` folder in the root of the repository
2. Run `docker compose up` command

The service consists of four containers - model-preparation, inference, database, api

### Model preparation
This container handles model conversion to onnx format

### Inference
This container deploys the newly created onnx model to the triton inference server.
This container only starts when the `model-preparation` container successfully finishes its work

### Database
This container hosts a PostgreSQL server for storing the user ids

#### API
This container exposes the `/predict` endpoint to the outer world, so we can submit a POST request with a JSON that contains `user_id` and `text` fields.
This container only starts after `inference` and `database` are considered as healthy

Example:

```
curl -X POST http://localhost:5000/predict \
   -H 'Content-Type: application/json' \
   -d '{"user_id":1, "text": "57 year old man with pancreatitis, alcohol withdrawal, tachypnea"}'
```
