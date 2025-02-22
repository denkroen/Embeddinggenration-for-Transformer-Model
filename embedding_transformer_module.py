import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from nn_transformer import TransformerEncoderM
from nn_none_wrapper import none_wrapper
from nn_baseline import baseline_CNN
from nn_mlp import MLP
import numpy as np
from utils import efficient_distance, reader_att_rep
from nn_gnn import GNN


from nn_tcnn import TCNN
from tcnn_autoencoder_module import TCNN_Autoencoder

class EmbeddingTransformer(pl.LightningModule):
    def __init__(self, learning_rate, num_filters, filter_size, mode, num_attributes, num_classes, window_length, sensor_channels, path_attributes, n_trans_layers, n_trans_hidden_neurons, embedding_model, pretrained, freeze, time_tokens, num_layers, embedding_size, checkpoint_path, learnable_adj=True, GSL=False):
        super().__init__()


        #TODO: rework training methods, add embedding rotation in generator modules
        self.time_tokens = time_tokens
        self.lr = learning_rate #def schedule
        self.embedding_model = embedding_model
        embedding_conv_channels = 1
        self.mode = mode

        if self.mode == "attribute":
            self.loss = nn.BCELoss()
            output_neurons = num_attributes

            # load attribute mapping
            self.attr = reader_att_rep(path_attributes) 
            for attr_idx in range(self.attr.shape[0]):
                self.attr[attr_idx, 1:] = self.attr[attr_idx, 1:] / np.linalg.norm(self.attr[attr_idx, 1:])

            self.atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)
            self.atts = self.atts.type(dtype=torch.cuda.FloatTensor)

        elif self.mode == "classification":
            self.loss = nn.CrossEntropyLoss()
            output_neurons = num_classes

        #embedding generator
        if embedding_model == "tcnn":

            #init attribute TCNN
            self.embedding_generator = TCNN(1,window_length,sensor_channels,filter_size,19,num_filters)
            embedding_conv_channels = 64

            if pretrained == True:
                #load weights
                checkpoint = torch.load(checkpoint_path)
                weights = checkpoint["state_dict"]
                #adjust keys from initialization
                weights = {k.replace('tcnn.', ''):weights[k] for k in weights}
                self.embedding_generator.load_state_dict(weights)
                


        elif embedding_model == "autoencoder":

            
            self.embedding_generator = TCNN_Autoencoder(1,window_length,sensor_channels,filter_size,num_classes,num_filters)
            embedding_conv_channels = 256


            if pretrained == True:
                #load weights
                checkpoint = torch.load(checkpoint_path)
                weights = checkpoint["state_dict"]
                #adjust keys from initialization
                weights = {k.replace('autoencoder.', ''):weights[k] for k in weights}
                self.embedding_generator.load_state_dict(weights)
        
        elif embedding_model == "gnn":
            #based on sensor channels reconstruct sensors
            if sensor_channels == 30: # mbientlab
                sensor_ch = 6
                num_sensors = 5
            elif sensor_channels == 27: # motionminers
                sensor_ch = 9
                num_sensors = 3
            elif sensor_channels == 126: # mocap
                sensor_ch = 6
                num_sensors = 21


            self.embedding_generator = GNN(window_length,embedding_size,num_layers,sensor_ch,num_sensors,learnable_adj)
            gls = True
            if gls == True:
                sensor_channels = num_sensors

            pass

        elif embedding_model == "mlp":
            self.embedding_generator = MLP(sensor_channels,window_length,num_layers, embedding_size)
            sensor_channels = embedding_size #the mlp applies in sensor space. 


            pass

        elif embedding_model == "none":
            self.embedding_generator = none_wrapper(sensor_channels, window_length)


        elif embedding_model == "baseline":
            self.embedding_generator = baseline_CNN(window_length,sensor_channels, embedding_size)
            sensor_channels = embedding_size #the convolutions happen in sensor space. 


        if freeze == True:
            self.embedding_generator.eval()

        embedding_length = self.embedding_generator.get_embedding_size()
            
        #transformer
        # if the time dimention should be the Token, we switch the dimensions and rotate the input in the forward pass
        if self.time_tokens:
            self.transformer = TransformerEncoderM(embedding_length, sensor_channels, n_trans_layers, n_trans_hidden_neurons)
        else:
            self.transformer = TransformerEncoderM(sensor_channels, embedding_length, n_trans_layers, n_trans_hidden_neurons)



        #classificator
        self.activation_function = nn.GELU()
        # Klassification token changes length based on time/sensor tokens
        if self.time_tokens:
            self.imu_head = nn.Sequential(nn.LayerNorm(embedding_length), nn.Linear(embedding_length, embedding_length//4),
                                        self.activation_function, nn.Dropout(0.1), nn.Linear(embedding_length//4, output_neurons))
        else:
            if sensor_channels == 3:
                self.imu_head = nn.Sequential(nn.LayerNorm(3),nn.Linear(3,3),self.activation_function, nn.Dropout(0.1), nn.Linear(3,output_neurons))

            else:
                self.imu_head = nn.Sequential(nn.LayerNorm(sensor_channels), nn.Linear(sensor_channels, sensor_channels//4),
                                            self.activation_function, nn.Dropout(0.1), nn.Linear(sensor_channels//4, output_neurons))
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()


        #self.loss = nn.CrossEntropyLoss()



        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average= "weighted"
        )

        self.embedding_conv = nn.Conv2d(in_channels=embedding_conv_channels,
                                        out_channels=1,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0)



    def forward(self, x):

        embedding, prediction_embedding_generator, reconst = self.embedding_generator.forward(x)

        if self.embedding_model == "tcnn" or self.embedding_model == "autoencoder": #needs an embedding convolution
            embedding = F.relu(self.embedding_conv(embedding))

        if self.embedding_generator != "baseline" or "gnn": #baseline cnn aready squeezes before the convolutions
            embedding = torch.squeeze(embedding,1) # important for the lara data. (Batch, 1, sensors, time) -> (Batch,sensors,time)

        if self.time_tokens: 
            embedding = embedding.permute(0, 2, 1) # rotate input to use time as tokens. this works, because The Transformer permutes the data again.


        x = self.transformer.forward(embedding)

        x = self.imu_head.forward(x)
        if self.mode == "attribute":
            x = F.sigmoid(x)
        else:
            x = F.softmax(x)




        return x, prediction_embedding_generator, reconst

    def training_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        train_acc = self.accuracy(prediction, label)
        train_f1 = self.f1(prediction, label)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": train_acc,
                "train_f1": train_f1
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        

        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        #calc_metrics
        val_acc = self.accuracy(prediction, label)
        val_f1 = self.f1(prediction, label)


        self.log_dict(
            {
                "validation_loss": loss,
                "validation_acc": val_acc,
                "validation_f1": val_f1
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return loss

    def test_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        test_acc = self.accuracy(prediction, label)
        test_f1 = self.f1(prediction, label)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": test_acc,
                "test_f1": test_f1

            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        
        input_tensor = batch["data"]
        label = batch["label_class"]
        label_attr = batch["label_attr"]

        pred, pred_embedding_generator, reconst = self.forward(input_tensor)

        #pred = torch.argmax(pred,1)

        #print (label)

        if self.mode == "attribute":
            loss = self.loss(pred, label_attr)
        else:
            loss = self.loss(pred, label) 



        if self.mode == "attribute":
            pred = pred.detach().clone()
            pred = efficient_distance(self.attr, self.atts, pred) #distances
            pred = self.atts[torch.argmin(pred, dim=1), 0] #classes

        return loss, pred, label
    
    

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        scheduler = scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}
