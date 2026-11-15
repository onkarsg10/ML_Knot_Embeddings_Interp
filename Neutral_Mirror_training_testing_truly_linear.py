import argparse
import os
from datetime import datetime
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from tqdm import tqdm

# Add missing imports
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import ast





###This is the method to split original data into train, val, test sets
def split_data_modified(df, group_size=37999, test_size=0.15, val_size=0.15, random_seed=42):
    # Create a RandomState with the fixed seed
    rng = np.random.RandomState(random_seed)

    # Step 1: Isolate the first group
    first_group = df.iloc[0:group_size]

    # Calculate split sizes
    val_count = int(val_size * group_size)
    test_count = int(test_size * group_size)

    # Perform random shuffle on indices of the first group using the fixed RandomState
    shuffled_indices = rng.permutation(first_group.index)

    # Split indices for train, val, and test
    val_indices = shuffled_indices[:val_count]
    test_indices = shuffled_indices[val_count:val_count + test_count]
    train_indices = shuffled_indices[val_count + test_count:]

    # Initialize empty sets for the overall split indices
    train_set, val_set, test_set = set(), set(), set()

    # Assign the first group indices to their respective sets
    for idx in train_indices:
        train_set.add(idx % group_size)
    for idx in val_indices:
        val_set.add(idx % group_size)
    for idx in test_indices:
        test_set.add(idx % group_size)

    # Step 2: Assign all other samples to the split of their corresponding index in the first group
    for idx in df.index:
        mod_idx = idx % group_size
        if mod_idx in train_set:
            train_set.add(idx)
        elif mod_idx in val_set:
            val_set.add(idx)
        elif mod_idx in test_set:
            test_set.add(idx)

    # Create the final DataFrames based on the split indices
    train_df = df.loc[df.index.isin(train_set)]
    val_df = df.loc[df.index.isin(val_set)]
    test_df = df.loc[df.index.isin(test_set)]

    return train_df, val_df, test_df

# # Apply the modified splitting function with a fixed random state
# train_df, val_df, test_df = split_data_modified(df, random_seed=42)






###This is the Braid Dataset and Braid Data Module WITHOUT flatteninig the tensors, so that after mirror operation I can flatten tensors myself
class BraidDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.braid_groups = {braid: list(sub_df.index) for braid, sub_df in self.df.groupby('Braid word')}
        self.braid_groups = {k: v for k, v in self.braid_groups.items() if len(v) > 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
          #max_len = 41 ### changed from 101 to 41 since now max braid length is 40 instead of 100 from befor
          max_len = 101 ### changed from 41 to 101
          pad_value = -100 ###I think pad value was -100

          # Safely convert string representations of lists into actual lists
          def parse_list(s):
              if isinstance(s, str):
                  return ast.literal_eval(s)
              return s

          # Get the braid word and formatted braid word for the current index
          braid_word = self.df.iloc[index]['Braid word']
          formatted_braid = parse_list(self.df.iloc[index]['Formatted braid word'])

          positive_indices = self.braid_groups[braid_word]
          positive_index = np.random.choice([idx for idx in positive_indices if idx != index])
          positive_braid = parse_list(self.df.iloc[positive_index]['Formatted braid word'])

          negative_braid_word = np.random.choice([bw for bw in self.braid_groups.keys() if bw != braid_word])
          negative_index = np.random.choice(self.braid_groups[negative_braid_word])
          negative_braid = parse_list(self.df.iloc[negative_index]['Formatted braid word'])

          # Pad sequences and convert to tensor
          # formatted_braid_tensor = torch.tensor([formatted_braid + [pad_value] * (max_len - len(formatted_braid))])
          # positive_braid_tensor = torch.tensor([positive_braid + [pad_value] * (max_len - len(positive_braid))])
          # negative_braid_tensor = torch.tensor([negative_braid + [pad_value] * (max_len - len(negative_braid))])


          # Convert to tensors without adding an unnecessary dimension
          formatted_braid_tensor = torch.tensor(formatted_braid + [pad_value] * (max_len - len(formatted_braid)), dtype=torch.float)
          positive_braid_tensor = torch.tensor(positive_braid + [pad_value] * (max_len - len(positive_braid)), dtype=torch.float)
          negative_braid_tensor = torch.tensor(negative_braid + [pad_value] * (max_len - len(negative_braid)), dtype=torch.float)


        ###Defining the new one hot encoding method
          def one_hot_encode_braid(braid_tensor, max_len=101):
              # Create an output tensor of zeros with shape [max_len, 62]
              # 62 = 30 (positive) + 30 (negative) + 1 (zero) + 1 (padding)
              encoded_tensor = torch.zeros((max_len, 62), dtype=torch.float)

              for idx in range(max_len):
                  value = int(braid_tensor[idx].item())
                  if value == -100:
                      encoded_tensor[idx, 61] = 1  # Padding value at the last position
                  elif value == 0:
                      encoded_tensor[idx, 60] = 1  # Zero at the second-to-last position
                  elif value > 0:
                      encoded_tensor[idx, value - 1] = 1  # Positive values 1 to 30 map to indices 0 to 29
                  elif value < 0:
                      encoded_tensor[idx, abs(value) + 29] = 1  # Negative values -1 to -30 map to indices 30 to 59

              return encoded_tensor

          oh_formatted_braid_tensor =  one_hot_encode_braid(formatted_braid_tensor, max_len=max_len)
          oh_positive_braid_tensor =  one_hot_encode_braid(positive_braid_tensor, max_len=max_len)
          oh_negative_braid_tensor =  one_hot_encode_braid(negative_braid_tensor, max_len=max_len)

          ####UNFLATTENING THE TENSOR
          # ohf_formatted_braid_tensor = oh_formatted_braid_tensor.flatten()
          # ohf_positive_braid_tensor = oh_positive_braid_tensor.flatten()
          # ohf_negative_braid_tensor = oh_negative_braid_tensor.flatten()




          return (oh_formatted_braid_tensor, oh_positive_braid_tensor, oh_negative_braid_tensor,
                  braid_word, self.df.iloc[positive_index]['Braid word'], self.df.iloc[negative_index]['Braid word'],
                  positive_index, negative_index)
          # return (ohf_formatted_braid_tensor, ohf_positive_braid_tensor, ohf_negative_braid_tensor,
          #         braid_word, self.df.iloc[positive_index]['Braid word'], self.df.iloc[negative_index]['Braid word'],
          #         positive_index, negative_index)

class BraidDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size=32, num_workers=4):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = BraidDataset(self.train_df)
            self.val_dataset = BraidDataset(self.val_df)
        if stage == 'test' or stage is None:
            self.test_dataset = BraidDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)






###This is the constrastive loss model (using feedforward model)
class BraidEncoderModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super(BraidEncoderModel, self).__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(6262, 512),  #####changed 17*41=697 to 101*32=3232
            nn.ReLU(),
            nn.Linear(512, 256),  # More neurons
            nn.ReLU(),
            nn.Linear(256, 128),  # More neurons
            nn.ReLU(),
            nn.Linear(128, 50)
        )

        self.validation_distances = []
        self.validation_labels = []


    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch
        anchor = self.forward(formatted_braids)
        positive = self.forward(positive_braids)
        negative = self.forward(negative_braids)
        positive_loss = F.pairwise_distance(anchor, positive)
        negative_loss = F.pairwise_distance(anchor, negative)
        loss = F.relu(positive_loss - negative_loss + 0.5).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch
        anchor = self.forward(formatted_braids)
        positive = self.forward(positive_braids)
        negative = self.forward(negative_braids)
        positive_loss = F.pairwise_distance(anchor, positive)
        negative_loss = F.pairwise_distance(anchor, negative)
        loss = F.relu(positive_loss - negative_loss + 0.5).mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        # Store results for epoch end calculation
        self.validation_distances.append({'positive_loss': positive_loss.detach(), 'negative_loss': negative_loss.detach()})

        return loss


    def on_validation_epoch_end(self):
        if self.validation_distances:  # Check if the list is not empty
            positive_distances = torch.cat([x['positive_loss'] for x in self.validation_distances])
            negative_distances = torch.cat([x['negative_loss'] for x in self.validation_distances])
            labels = torch.cat([torch.ones_like(positive_distances), torch.zeros_like(negative_distances)])
            distances = torch.cat([positive_distances, negative_distances])
            auc_score = roc_auc_score(labels.cpu().numpy(), -distances.cpu().numpy())
            self.log('val_auc', auc_score)
        else:
            print("No data collected for validation.")

        # Clear the list after processing to prepare for the next validation epoch
        self.validation_distances.clear()



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)





###This is the TINY classification model with a modification in forward method suited to test the mirror NN
class CLASSIFIER_BraidDualHeadModel(pl.LightningModule):
    def __init__(self, frozen_model):
        super(CLASSIFIER_BraidDualHeadModel, self).__init__()
        self.frozen_model = frozen_model
        for param in self.frozen_model.parameters():
            param.requires_grad = False  # Freeze the parameters

        ####This is the one I tried first with a single hidden layer with 50 neurons; wandb: dashing tree 36
        # self.classifier = nn.Sequential(
        #     nn.Linear(100, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 1),
        #     nn.Sigmoid()
        # )


        # self.classifier = nn.Sequential( ####OG small classifier
        #     nn.Linear(100, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 1),
        #     nn.Sigmoid()
        # )
        self.classifier = nn.Sequential( ####Bigger classifier
            nn.Linear(100, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )


    def forward(self, input1, input2): ##############Changed forward method a lot since got rid of code running frozen model here since for testing mirror, we want to run classsifier directly on input1, input2, since due to problem setup they both are already in the embedding space
        # self.frozen_model.eval()  # Ensure the model is in eval mode
        # with torch.no_grad():  # Ensure no gradients are computed for frozen_model
        #     encoded1 = self.frozen_model(input1)
        #     encoded2 = self.frozen_model(input2)
        #combined_features = torch.cat((encoded1, encoded2), dim=1)
        combined_features = torch.cat((input1, input2), dim=1) ##############Made this change + got rid of code running frozen model here since for testing mirror, we want to run classsifier directly on input1, input2, since due to problem setup they both are already in the embedding space
        return self.classifier(combined_features)

    def compute_accuracy(self, predictions, labels):
        predictions = predictions.round()  # Convert probabilities to 0 or 1
        correct_count = (predictions == labels).float().sum()  # Count correct predictions
        return correct_count / len(predictions)  # Compute accuracy

    def training_step(self, batch, batch_idx):
        formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch
        pos_pred = self.forward(formatted_braids, positive_braids)
        neg_pred = self.forward(formatted_braids, negative_braids)
        labels = torch.cat((torch.ones_like(pos_pred), torch.zeros_like(neg_pred)), dim=0)
        predictions = torch.cat((pos_pred, neg_pred), dim=0)
        loss = F.binary_cross_entropy(predictions, labels)
        acc = self.compute_accuracy(predictions, labels)  # Compute accuracy
        self.log('train_loss', loss)
        self.log('train_acc', acc)  # Log accuracy

        return loss

    def validation_step(self, batch, batch_idx):
        formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch
        pos_pred = self.forward(formatted_braids, positive_braids)
        neg_pred = self.forward(formatted_braids, negative_braids)
        labels = torch.cat((torch.ones_like(pos_pred), torch.zeros_like(neg_pred)), dim=0)
        predictions = torch.cat((pos_pred, neg_pred), dim=0)
        loss = F.binary_cross_entropy(predictions, labels)
        acc = self.compute_accuracy(predictions, labels)  # Compute validation accuracy
        self.log('val_loss', loss)
        self.log('val_acc', acc)  # Log validation accuracy

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)







#####This is the mirror training model
class BraidDualHeadModel(pl.LightningModule):
    def __init__(self, frozen_model, learning_rate, args):
        super(BraidDualHeadModel, self).__init__()
        self.frozen_model = frozen_model
        self.learning_rate = learning_rate  # Store the learning rate
        # Freeze the parameters of the frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        ########bias = 0 allows for MATHEMATICALLY LINEAR FUNCTION
        use_bias = bool(args.bias)
        self.classifier = nn.Sequential(
        nn.Linear(50, 10, bias=use_bias),
            nn.Linear(10, 50, bias=use_bias)
        )


    # Define the modified function M for the smaller example
    # def M(self,encoded_tensor):
    #     result_tensor = encoded_tensor.clone()
    #     # Modify the first 5 entries in the last dimension for all items in the batch and all steps
    #     result_tensor[:, :, :30] *= -1
    #     return result_tensor

    ####Defining the new mirror method 
    def M(self, encoded_tensor):
        result_tensor = encoded_tensor.clone()
        # Swap the positive and negative sections
        result_tensor[:, :, :30], result_tensor[:, :, 30:60] = result_tensor[:, :, 30:60].clone(), result_tensor[:, :, :30].clone()
        return result_tensor


    def forward(self, input1, input2):
        self.frozen_model.eval()
        with torch.no_grad():

            ####OLD SYNTAX FOR RESHAPE
            mirrored_encoded1_old_syntax = self.frozen_model(self.M(input1).reshape(-1,101*62))
            mirrored_encoded2_old_syntax = self.frozen_model(self.M(input2).reshape(-1,101*62))
            #unmirrored_encoded2_old_syntax= self.frozen_model(input2) ###EASY VERSION OF GAME SINCE KNOT DIAG IS SAME
            unmirrored_encoded2_old_syntax= self.frozen_model(input1.reshape(-1,101*62)) #### HARD VERSION OF GAME


            ###NEW SYNTAX FOR RESHAPING (THIS IS THE ONE WE ARE ACTUALLY USING IN OUR COMPUTATION; OLD ONE IS HERE JUST FOR SANITY CHECK TO BE SURE THEY ARE BOTH EQUAL)
            mirrored_encoded1 = self.frozen_model(self.M(input1).view(self.M(input1).size(0),-1))
            mirrored_encoded2 = self.frozen_model(self.M(input2).view(self.M(input2).size(0),-1))
            #unmirrored_encoded2= self.frozen_model(input2) ###EASY VERSION OF GAME SINCE KNOT DIAG IS SAME
            unmirrored_encoded2= self.frozen_model(input1.view(input1.size(0),-1)) #### HARD VERSION OF GAME

            if(torch.equal(mirrored_encoded1_old_syntax, mirrored_encoded1) and torch.equal(mirrored_encoded2_old_syntax, mirrored_encoded2) and torch.equal(unmirrored_encoded2_old_syntax, unmirrored_encoded2)):
                #print("EQUAL")
                pass
            else:
                print("not equal")

        predicted_mirror = self.classifier(unmirrored_encoded2)
        return predicted_mirror, mirrored_encoded2

    def euclidean_distance(self, x, y):
        return torch.norm(x - y, p=2, dim=1)  # Compute L2 norm along dimension 1

    def training_step(self, batch, batch_idx):
        formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch
        pred_mirror, mirrored_positive = self.forward(formatted_braids, positive_braids)
        loss = F.mse_loss(pred_mirror, mirrored_positive)
        # print("pred shape")
        # print(pred_mirror.shape)
        # print("actual shape")
        # print(mirrored_positive.shape)
        distance = self.euclidean_distance(pred_mirror, mirrored_positive)
        self.log('train_loss', loss)
        self.log('train_distance', distance.mean())  # Log the average Euclidean distance
        return loss

    def validation_step(self, batch, batch_idx):
        formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch
        pred_mirror, mirrored_positive = self.forward(formatted_braids, positive_braids)
        loss = F.mse_loss(pred_mirror, mirrored_positive)
        # print("pred shape")
        # print(pred_mirror.shape)
        # print("actual shape")
        # print(mirrored_positive.shape)
        distance = self.euclidean_distance(pred_mirror, mirrored_positive)
        self.log('val_loss', loss)
        self.log('val_distance', distance.mean())  # Log the average Euclidean distance
        return {'val_loss': loss, 'val_distance': distance.mean()}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)







###This is the main method to run the training
def main(args):
    # Record when this run started (Colab runtimes use UTC); appended to checkpoint filenames
    run_start = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Seed Python, NumPy and PyTorch (including DataLoader workers) so the same command gives the same run
    pl.seed_everything(args.seed, workers=True)

    # Set up wandb login
    wandb.login(key="<enter wandb key>")

    # Set base path based on whether we're in Colab or not
    base_path = '/content/drive/My Drive/' if args.is_colab else ''

    # Load and split the data
    df = pd.read_csv(os.path.join(base_path, args.csv_path))
    train_df, val_df, test_df = split_data_modified(df, random_seed=42)

    # Initialize the data module
    data_module = BraidDataModule(train_df, val_df, test_df, batch_size=args.batch_size, num_workers=4)

    # Initialize the frozen model
    frozen_model = BraidEncoderModel(learning_rate=args.learning_rate)

    # Load the checkpoint for the frozen model
    frozen_checkpoint = torch.load(args.frozen_model_checkpoint)
    frozen_model.load_state_dict(frozen_checkpoint['state_dict'])

    # Initialize the classifier model
    classifier_model = CLASSIFIER_BraidDualHeadModel(frozen_model)

    # Initialize the mirror model
    mirror_model = BraidDualHeadModel(frozen_model, args.learning_rate, args)

    if args.test:
        if args.mirror_checkpoint is None or args.classifier_checkpoint is None:
            raise ValueError("Mirror and classifier checkpoints are required for testing mode.")
        
        # Load the mirror model checkpoint
        mirror_checkpoint = torch.load(args.mirror_checkpoint)
        mirror_model.load_state_dict(mirror_checkpoint['state_dict'])

        # Load the classifier model checkpoint
        classifier_checkpoint = torch.load(args.classifier_checkpoint)
        classifier_model.load_state_dict(classifier_checkpoint['state_dict'])

        # Run test mode
        test_model(mirror_model, classifier_model, frozen_model, data_module)
    else:
        # Configure WandbLogger for training
        wandb_logger = WandbLogger(project='Braid - Mirror Experiments', name='Training Run')
        wandb_logger.experiment.config.update(vars(args))

        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(base_path, args.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Configure ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            monitor='val_distance',
            dirpath=checkpoint_dir,
            filename=f'Neutral_Mirror_best-checkpoint-{{epoch:02d}}-{{val_distance:.5f}}-lr{args.learning_rate:.5e}-vci{args.val_check_interval:.3f}-vf{args.val_fraction:.3f}-bs{args.batch_size}-bias{args.bias}-start{run_start}',
            save_top_k=1,
            mode='min',
            save_weights_only=True
        )

        # Configure Trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=args.epochs,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.val_fraction,
            devices=1,
            accelerator="auto",
            deterministic=True,
            callbacks=[checkpoint_callback],
        )

        # Fit the model
        trainer.fit(mirror_model, data_module)

        # Close the Wandb run
        wandb.finish()

def test_model(mirror_model, classifier_model, frozen_model, data_module):
    # Initialize wandb for test run
    wandb.init(project='Braid - Mirror Experiments', name='Test Run', job_type='evaluation')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set models to evaluation mode
    classifier_model.eval()
    mirror_model.eval()
    frozen_model.eval()

    classifier_model = classifier_model.to(device)
    frozen_model = frozen_model.to(device)
    mirror_model = mirror_model.to(device)

    # Set up the test dataset
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()

    with torch.no_grad():
        total_accuracy = 0
        total_loss = 0
        total_samples = 0
        total_pos_accuracy = 0
        total_pos_samples = 0
        total_neg_accuracy = 0
        total_neg_samples = 0
        total_orig_accuracy = 0
        total_orig_samples = 0

        progress = tqdm(test_dataloader, total=len(test_dataloader), desc='Evaluating', leave=True)

        for batch in progress:
            formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch

            formatted_braids = formatted_braids.to(device)
            positive_braids = positive_braids.to(device)
            negative_braids = negative_braids.to(device)

            pos_predicted_mirrored_formatted, pos_actual_mirrored_pos = mirror_model(formatted_braids, positive_braids)
            neg_predicted_mirrored_formatted, neg_actual_mirrored_neg = mirror_model(formatted_braids, negative_braids)

            pos_pred = classifier_model(pos_predicted_mirrored_formatted, pos_actual_mirrored_pos)
            neg_pred = classifier_model(neg_predicted_mirrored_formatted, neg_actual_mirrored_neg)

            pos_pred = pos_pred.squeeze()
            neg_pred = neg_pred.squeeze()

            labels = torch.cat((torch.ones(pos_pred.size(0), device=device), torch.zeros(neg_pred.size(0), device=device)), dim=0)
            predictions = torch.cat((pos_pred, neg_pred), dim=0)

            loss = F.binary_cross_entropy(predictions, labels)
            total_loss += loss.item() * (pos_pred.size(0) + neg_pred.size(0))

            accuracy = classifier_model.compute_accuracy(predictions, labels)
            total_accuracy += accuracy.item() * (pos_pred.size(0) + neg_pred.size(0))
            total_samples += pos_pred.size(0) + neg_pred.size(0)

            pos_accuracy = classifier_model.compute_accuracy(pos_pred, torch.ones(pos_pred.size(0), device=device))
            total_pos_accuracy += pos_accuracy.item() * pos_pred.size(0)
            total_pos_samples += pos_pred.size(0)

            neg_accuracy = classifier_model.compute_accuracy(neg_pred, torch.zeros(neg_pred.size(0), device=device))
            total_neg_accuracy += neg_accuracy.item() * neg_pred.size(0)
            total_neg_samples += neg_pred.size(0)

            # Original-knot pairs C(L(f(x)), f(y)): same x and y as the positive pairs, but y is not mirrored
            actual_unmirrored_pos = frozen_model(positive_braids.view(positive_braids.size(0), -1))
            orig_pred = classifier_model(pos_predicted_mirrored_formatted, actual_unmirrored_pos).squeeze()
            orig_accuracy = classifier_model.compute_accuracy(orig_pred, torch.zeros(orig_pred.size(0), device=device))
            total_orig_accuracy += orig_accuracy.item() * orig_pred.size(0)
            total_orig_samples += orig_pred.size(0)

            progress.set_postfix(loss=loss.item(), accuracy=accuracy.item(), pos_accuracy=pos_accuracy.item())

            # Log batch metrics to wandb
            wandb.log({
                'batch_loss': loss.item(),
                'batch_accuracy': accuracy.item(),
                'batch_pos_accuracy': pos_accuracy.item()
            })

        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples
        average_pos_accuracy = total_pos_accuracy / total_pos_samples
        average_neg_accuracy = total_neg_accuracy / total_neg_samples
        average_orig_accuracy = total_orig_accuracy / total_orig_samples

        print(f"Test Loss: {average_loss:.6f}")
        print(f"Test Accuracy: {average_accuracy:.6f}")
        print(f"Positive Predictions Accuracy: {average_pos_accuracy:.6f}")

        # Readable summary: notation first, then what each check shows and its result
        bias_note = "with bias" if mirror_model.classifier[0].bias is not None else "no bias"
        summary = [
            "",
            "======================== Mirror map test: results ========================",
            "Notation",
            "  K       a knot from the test split (never seen when training f, L or C);",
            "          every test diagram of every test knot is used in turn",
            "  x       a braid diagram of K",
            "  y       another diagram of the same knot K, picked at random",
            "  z       a diagram of a different knot, picked at random",
            "  M       the mirror image of a braid diagram (every crossing flipped: each",
            "          letter i in the braid word becomes -i), so M(y) is a diagram of",
            "          the mirror of K",
            "  f       the frozen contrastive encoder: braid diagram -> 50-dim embedding",
            "  L       the learnt linear mirror map on embeddings (50 -> 10 -> 50,",
            f"          {bias_note}), trained so that L(f(x)) is close to f(M(y));",
            "          L(f(x)) is its prediction of the embedding of the mirror of K",
            '  C(a, b) the pretrained classifier; an output above 0.5 means "a and b are',
            '          the same knot"',
            "",
            '1. Positive pairs C(L(f(x)), f(M(y))): want "same", higher is better',
            f"   Result: {average_pos_accuracy:.6f} over {total_pos_samples} diagrams",
            "",
            '2. Negative pairs C(L(f(x)), f(M(z))): want "different", higher is better',
            f"   Result: {average_neg_accuracy:.6f} over {total_neg_samples} diagrams",
            "",
            '3. Original-knot pairs C(L(f(x)), f(y)): want "different", higher is better',
            f"   Result: {average_orig_accuracy:.6f} over {total_orig_samples} diagrams",
            "==========================================================================",
        ]
        print("\n".join(summary))

        # Log final test metrics to wandb
        wandb.log({
            'test_loss': average_loss,
            'test_accuracy': average_accuracy,
            'test_pos_accuracy': average_pos_accuracy,
            'test_neg_accuracy': average_neg_accuracy,
            'test_orig_knot_rejection': average_orig_accuracy
        })

    # Close the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a mirror model for braid classification")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--val_check_interval", type=float, default=0.05, help="Fraction of epoch to run validation")
    parser.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of validation set to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--is_colab", type=int, default=0, help="1 if running in Google Colab, 0 otherwise")
    parser.add_argument("--checkpoint_path", type=str, default='Mirror_model_checkpoints', help="Path to save checkpoints")
    parser.add_argument("--frozen_model_checkpoint", type=str, required=True, 
                        help="Path to the checkpoint file for the frozen model")
    parser.add_argument("--test", type=int, default=0, help="1 to run in test mode, 0 for training mode")
    parser.add_argument("--mirror_checkpoint", type=str, default=None, help="Path to the mirror model checkpoint for testing")
    parser.add_argument("--classifier_checkpoint", type=str, default=None, help="Path to the classifier model checkpoint for testing")
    parser.add_argument('--bias', type=int, default=1, choices=[0, 1], help='Use bias in linear layers (0: False, 1: True)')    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)