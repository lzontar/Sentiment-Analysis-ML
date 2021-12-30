import gc
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.autograd.grad_mode import F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from sklearn import metrics
from util import helper
from util.BertClassifierInner import BertClassifierInner
from util.helper import set_seed
from util.models.IClassifier import IClassifier


class BERT_(IClassifier):
    def __init__(self, n_splits=10, max_length=64, batch_size=32, epochs=4, primary=False):
        self.primary = primary
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate Bert Classifier
        self.model = BertClassifierInner(freeze_bert=False)
        # Tell PyTorch to run the model on GPU
        self.model.to(self.device)

        # Create the optimizer
        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )

        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        self.loss_fn = nn.CrossEntropyLoss()
        self.label = 'BERT'
        self.n_splits = n_splits
        self.random_state = 42
        self.batch_size = batch_size
        self.max_length = max_length
        self.epochs = epochs

    def prepDataClassifier(self, X_train, y_train, X_test, y_test):

        # Run function `preprocessing_for_bert` on the train set and the validation set
        train_inputs, train_masks = self.preprocessing_for_bert(X_train)
        test_inputs, test_masks = self.preprocessing_for_bert(X_test)

        # Convert other data types to torch.Tensor
        train_labels = torch.tensor(y_train.values)
        test_labels = torch.tensor(y_test.values)

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        # Create the DataLoader for our validation set
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

        return train_dataloader, test_dataloader

    def fitClassifierInner(self, X_train, y_train, X_test, y_test):
        train_dataloader, test_dataloader = self.prepDataClassifier(X_train, y_train, X_test, y_test)

        # Total number of training steps
        total_steps = len(train_dataloader) * self.epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=total_steps)
        set_seed(self.random_state)  # Set seed for reproducibility
        self.train(train_dataloader, self.epochs, test_dataloader, evaluation=True)

        self.save()

    def predict(self, X, y):
        """Perform a forward pass on the trained BERT model to predict probabilities
            on the test set.
            """

        # Run function `preprocessing_for_bert` on the train set and the validation set
        train_inputs, train_masks = self.preprocessing_for_bert(X)

        # Convert other data types to torch.Tensor
        train_labels = torch.tensor(y.values)

        # Create the DataLoader for our training set
        data = TensorDataset(train_inputs, train_masks, train_labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        preds, _, _ = self.evaluate(dataloader)
        return list(map(lambda x: int(x.item()), preds))

    def train(self, train_dataloader, epochs, test_dataloader=None, evaluation=False):
        """Train the BertClassifier model.
        """
        # Start training loop
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                _, test_loss, test_accuracy = self.evaluate(test_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {test_loss:^10.6f} | {test_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

    def evaluate(self, test_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        # Tracking variables
        test_accuracy = []
        test_loss = []

        predicted = []

        # For each batch in our validation set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            test_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            predicted.extend(preds)
            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            test_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        test_loss = np.mean(test_loss)
        test_accuracy = np.mean(test_accuracy)

        return predicted, test_loss, test_accuracy

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_length,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def save(self):
        if self.primary:
            torch.save(self.model.state_dict(), f'../data/results/models/BERT.h5')

    def load(self):
        self.model.load_state_dict(torch.load(f'../data/results/models/BERT.h5'))
