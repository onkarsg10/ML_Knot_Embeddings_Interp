import argparse
import os
from datetime import datetime
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

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

###This is the Braid Dataset and Braid Data Module
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


          ohf_formatted_braid_tensor = oh_formatted_braid_tensor.flatten()
          ohf_positive_braid_tensor = oh_positive_braid_tensor.flatten()
          ohf_negative_braid_tensor = oh_negative_braid_tensor.flatten()


          # return (formatted_braid_tensor, positive_braid_tensor, negative_braid_tensor,
          #         braid_word, self.df.iloc[positive_index]['Braid word'], self.df.iloc[negative_index]['Braid word'],
          #         positive_index, negative_index)

          return (ohf_formatted_braid_tensor, ohf_positive_braid_tensor, ohf_negative_braid_tensor,
                  braid_word, self.df.iloc[positive_index]['Braid word'], self.df.iloc[negative_index]['Braid word'],
                  positive_index, negative_index)

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
            nn.Linear(6262, 512),  # Updated input dimension
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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

    # Initialize the model
    model = BraidEncoderModel(learning_rate=args.learning_rate)

    # Configure WandbLogger
    wandb_logger = WandbLogger(project='Braid - Contrastive Loss Experiments')
    wandb_logger.experiment.config.update(vars(args))

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(base_path, args.checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Configure ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        dirpath=checkpoint_dir,
        #filename='Contrastive_best-checkpoint-{epoch:02d}-{val_auc:.2f}-lr{lr:.1e}-vci{vci:.2f}-vf{vf:.2f}-bs{bs}',
        #filename='Contrastive_best-checkpoint-{epoch:02d}-{val_auc:.5f}-lr{args.learning_rate:.5e}-vci{args.val_check_interval:.3f}-vf{args.val_fraction:.3f}-bs{args.batch_size}''
        filename=f'Neutral_Contrastive_best-checkpoint-{{epoch:02d}}-{{val_auc:.2f}}-lr{args.learning_rate:.1e}-vci{args.val_check_interval:.2f}-vf{args.val_fraction:.2f}-bs{args.batch_size}-start{run_start}',
        save_top_k=1,
        mode='max',
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
    trainer.fit(model, data_module)

    # Close the Wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a contrastive model for braid classification")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--val_check_interval", type=float, default=0.05, help="Fraction of epoch to run validation")
    parser.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of validation set to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--is_colab", type=int, default=0, help="1 if running in Google Colab, 0 otherwise")
    parser.add_argument("--checkpoint_path", type=str, default='Contrastive_model_checkpoints', help="Path to save checkpoints")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)