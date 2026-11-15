import argparse
import os
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
from tqdm import tqdm





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



##### This is the TINY classification model that uses embeddings from pretrained contrastive model
class BraidDualHeadModel(pl.LightningModule):
    def __init__(self, frozen_model, learning_rate):
        super(BraidDualHeadModel, self).__init__()
        self.frozen_model = frozen_model
        for param in self.frozen_model.parameters():
            param.requires_grad = False  # Freeze the parameters

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
        self.learning_rate = learning_rate  # Store learning rate

    def forward(self, input1, input2): #######Changed forward method from the classifier training model, by not applying frozen model to the data here, since the data that comes in here when we run the tests as we currently are, does not need any to be converted into a 50 dim embedding, it already comes that way from the test method where we are calling the forward method on the classifier model
        # self.frozen_model.eval()  # Ensure the model is in eval mode
        # with torch.no_grad():  # Ensure no gradients are computed for frozen_model
        #     encoded1 = self.frozen_model(input1)
        #     encoded2 = self.frozen_model(input2)
        # combined_features = torch.cat((encoded1, encoded2), dim=1)
        # return self.classifier(combined_features)
        combined_features = torch.cat((input1, input2), dim=1)
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)






###This is the test method
def test_model(model, frozen_model, data_module, device, wandb_logger):
    model.eval()
    frozen_model.eval()

    model = model.to(device)
    frozen_model = frozen_model.to(device)

    test_dataloader = DataLoader(data_module.test_dataset, batch_size=256, shuffle=False)

    # def M(encoded_tensor):
    #     result_tensor = encoded_tensor.clone()
    #     result_tensor[:, :, :30] *= -1
    #     return result_tensor

    #Defgining the new mirror method
    def M(encoded_tensor):
        result_tensor = encoded_tensor.clone()
        # Swap the positive and negative sections
        result_tensor[:, :, :30], result_tensor[:, :, 30:60] = result_tensor[:, :, 30:60].clone(), result_tensor[:, :, :30].clone()
        return result_tensor

    with torch.no_grad():
        total_accuracy = 0
        total_loss = 0
        total_samples = 0
        total_pos_accuracy = 0
        total_pos_samples = 0

        progress = tqdm(test_dataloader, total=len(test_dataloader), desc='Evaluating', leave=True)

        for batch in progress:
            formatted_braids, positive_braids, negative_braids, _, _, _, _, _ = batch

            formatted_braids = formatted_braids.to(device)
            positive_braids = positive_braids.to(device)
            negative_braids = negative_braids.to(device)

            # print("formatted_braids shape:", formatted_braids.shape)
            # print("positive_braids shape:", positive_braids.shape)
            # print("negative_braids shape:", negative_braids.shape)

            # print("formatted_braids shape modified:", (formatted_braids.view(formatted_braids.size(0),-1)).shape)
            # print("positive_braids shape modified:", (positive_braids.view(positive_braids.size(0),-1)).shape)
            # print("negative_braids shape modified:", (negative_braids.view(negative_braids.size(0),-1)).shape)

            pos_pred = model(frozen_model(formatted_braids.view(formatted_braids.size(0),-1)), 
                             frozen_model(M(positive_braids).view(M(positive_braids).size(0),-1)))
            neg_pred = model(frozen_model(formatted_braids.view(formatted_braids.size(0),-1)), 
                             frozen_model(M(formatted_braids).view(M(formatted_braids).size(0),-1)))
            
            x = frozen_model(formatted_braids.view(formatted_braids.size(0),-1))
            y = frozen_model(M(positive_braids).view(M(positive_braids).size(0),-1))

            distances = torch.norm(x - y, dim=1)
            average_distance = distances.mean().item()

            pos_pred = pos_pred.squeeze()
            neg_pred = neg_pred.squeeze()

            labels = torch.cat((torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))), dim=0).to(device)
            predictions = torch.cat((pos_pred, neg_pred), dim=0)

            loss = F.binary_cross_entropy(predictions, labels)
            total_loss += loss.item() * (pos_pred.size(0) + neg_pred.size(0))

            accuracy = model.compute_accuracy(predictions, labels)
            total_accuracy += accuracy.item() * (pos_pred.size(0) + neg_pred.size(0))
            total_samples += pos_pred.size(0) + neg_pred.size(0)

            pos_accuracy = model.compute_accuracy(pos_pred, torch.ones(pos_pred.size(0)).to(device))
            total_pos_accuracy += pos_accuracy.item() * pos_pred.size(0)
            total_pos_samples += pos_pred.size(0)

            progress.set_postfix(loss=loss.item(), accuracy=accuracy.item(), 
                                 pos_accuracy=pos_accuracy.item(), avg_dist=average_distance)

        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples
        average_pos_accuracy = total_pos_accuracy / total_pos_samples

        print(f"Test Loss: {average_loss:.6f}")
        print(f"Test Accuracy: {average_accuracy:.6f}")
        print(f"Test Pos Accuracy: {average_pos_accuracy:.6f}")

        # Replace the wandb_logger.log() call with the following:
        wandb.log({
            "test_loss": average_loss,
            "test_accuracy": average_accuracy,
            "test_pos_accuracy": average_pos_accuracy
        })







###This is the main method to run the training/testing
def main(args):
    # Seed Python, NumPy and PyTorch (including DataLoader workers) so the same command gives the same run
    pl.seed_everything(args.seed, workers=True)

    # Set up wandb login
    wandb.login(key="<enter wandb key>")

    # Initialize a new wandb run
    wandb.init(project='Braid - Classification Experiments', job_type='test', config=vars(args))

    # Set base path based on whether we're in Colab or not
    base_path = '/content/drive/My Drive/' if args.is_colab else ''

    # Load and split the data
    df = pd.read_csv(os.path.join(base_path, args.csv_path))
    train_df, val_df, test_df = split_data_modified(df, random_seed=42)

    # Initialize the data module
    data_module = BraidDataModule(train_df, val_df, test_df, batch_size=args.batch_size, num_workers=4)
    data_module.setup(stage='test')

    # Initialize the frozen model
    frozen_model = BraidEncoderModel(learning_rate=args.learning_rate)

    # Load the checkpoint for the frozen model
    checkpoint = torch.load(args.frozen_model_checkpoint)
    frozen_model.load_state_dict(checkpoint['state_dict'])

    # Instantiate the TINY classifier model
    model = BraidDualHeadModel(frozen_model, learning_rate=args.learning_rate)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    frozen_model = frozen_model.to(device)

    # Configure WandbLogger (keep this for compatibility with other parts of your code)
    wandb_logger = WandbLogger(project='Braid - Classification Experiments', job_type='test')

    if args.classifier_checkpoint is None:
        raise ValueError("Classifier checkpoint must be provided for testing.")
    
    # Load the classifier checkpoint
    checkpoint = torch.load(args.classifier_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Run the test
    test_model(model, frozen_model, data_module, device, wandb_logger)

    # Close the Wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a contrastive model for braid classification")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--is_colab", type=int, default=0, help="1 if running in Google Colab, 0 otherwise")
    parser.add_argument("--frozen_model_checkpoint", type=str, required=True, 
                        help="Path to the checkpoint file for the frozen model")
    parser.add_argument("--classifier_checkpoint", type=str, required=True, 
                        help="Path to the checkpoint file for the classifier model")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (used for model initialization)")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)