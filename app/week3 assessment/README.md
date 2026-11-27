Week 3 Assessment - Deep Learning + NLP
Customer feedback classification: from a small NumPy network to a fine-tuned DistilBERT served with FastAPI.

How to run (in this order)
pip install numpy pandas matplotlib scikit-learn torch transformers gensim fastapi uvicorn
python task1_nn_fundamentals.py
python task2_pytorch_pipeline.py
python task3_nlp_baseline.py      # creates data/train.csv and data/test.csv
python task4_finetune.py          # fine-tunes DistilBERT, saves models/ and comparison_table.md
python -m uvicorn task4_api:app --reload    # then open http://127.0.0.1:8000/docs
Model weights (models/, best_mlp.pt) are not committed; run the scripts to regenerate them.

Task 1 - Neural network fundamentals
2-4-1 NumPy network (ReLU hidden, Sigmoid output), binary cross-entropy, 5000 epochs, lr=0.5, learns XOR.
Gradient of (w*x + b - t)^2 at w=1, x=2, b=0, t=10: by hand dL/dw = 2(wx+b-t)x = -32, dL/db = 2(wx+b-t) = -16. PyTorch autograd gives the same values.
Loss curves (task1_loss_curves.png):
lr=0.01: learns very slowly, loss is still high after 5000 epochs.
lr=0.5: converges, loss close to 0.
lr=5: too large, training fails and the loss stays at about 0.693 (all predictions 0.5).
Dropout: Randomly switches off a fraction of neurons in each training step, so the network cannot depend on one neuron. It reduces overfitting and is turned off at inference (model.eval()).

BatchNorm: Normalises a layer's inputs over the mini-batch, then rescales with learnable parameters. It makes training more stable and faster and allows higher learning rates.

Early stopping: Watches validation loss/accuracy and stops when it has not improved for N epochs. It prevents overfitting and keeps the best weights.

Task 2 - PyTorch training pipeline
Breast cancer dataset, features scaled (scaler fitted on train only), CsvDataset(Dataset), DataLoaders with batch_size=32.
MLP(nn.Module) with Linear, BatchNorm1d, ReLU, Dropout. Parameters: 4,289.
Adam, 40 epochs, early stopping (patience 8). Early stopping triggered at epoch 29. Best validation accuracy 0.98 (0.9825) at epoch 21.
state_dict saved, loaded into a new model in eval mode: accuracy 0.9825 (identical).
Scikit-learn logistic regression accuracy: 0.9825, the same as the MLP. On this small, easy tabular dataset a linear model is already enough.
Task 3 - Classical NLP baseline and embeddings
Dataset: 600 synthetic, template-generated customer feedback texts (3 classes: positive / negative / neutral, 200 each), created by the script in task3_nlp_baseline.py and saved to data/feedback.csv. It is NOT real customer data (see the note below).
Cleaning: lowercase, punctuation removal, tokenisation. Same stratified 80/20 split saved to data/train.csv and data/test.csv and reused in Task 4.
TF-IDF + LogisticRegression: accuracy 1.00, macro F1 1.00 (confusion matrix in task3_confusion_matrix.png).
GloVe (glove-wiki-gigaword-50): king - man + woman -> queen (similarity 0.85). 5 nearest words to good: better, really, always, sure, something. To delivery: deliveries, cents, fast, dropped, dropping. Averaged word vectors with logistic regression: accuracy 0.992, macro F1 0.992.
Observation: TF-IDF (1.000) is slightly ahead of averaged GloVe vectors (0.992). Averaging loses word order and ignores words missing from GloVe, while TF-IDF uses the exact vocabulary of this dataset.
Important note: a perfect 1.00 is not realistic. The texts are generated from a small set of templates, so each class has very distinctive words (for example "hated", "loved", "called") and the test set shares the same patterns as the training set. This result shows that the pipeline works, not that sentiment classification is solved. Real customer feedback would give lower scores.
Task 4 - Transformers, fine-tuning and serving
DistilBERT tokenizer output, sentiment-analysis and ner pipelines are shown in task4_finetune.py.
distilbert-base-uncased fine-tuned with the Trainer API (3 epochs) on the same train/test split as the baseline.
Comparison (same test set):

Model	Accuracy	Macro F1	Training time	Inference time (120 texts)
TF-IDF + LogReg	1.000	1.000	0.37 s	11.0 ms
DistilBERT (fine-tuned, CPU)	1.000	1.000	94.7 s	1620 ms

Conclusion: On this dataset both models reach 1.000 accuracy and macro F1, so DistilBERT gives no measurable accuracy gain, but it needs about 250x more training time and about 150x more inference time on CPU. Because the data is synthetic and very easy, this only shows that the task is easy; it does not prove that TF-IDF is as good as DistilBERT in general. On real feedback with negation, sarcasm and varied wording, I would expect DistilBERT to do better, and I would test that on a real dataset before choosing a model.

API: POST /predict loads the model once and returns label and confidence; empty text returns 422.
Screenshots: screenshots/ (200 response and 422 response).

BERT (encoder) vs GPT (decoder):

BERT is an encoder: it reads the whole sentence in both directions, so it understands context well.
GPT is a decoder: it reads left to right and is built to generate the next token.
BERT is pre-trained by predicting masked words; GPT by predicting the next word.
For classification, a BERT-style encoder with a small classification head is smaller, faster and cheaper.
I would use DistilBERT for this task; GPT-style models suit text generation and chat.
