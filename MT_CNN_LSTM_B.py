import torch
import torch.nn as nn


class CNN1D_LSTM(nn.Module):

    def __init__(self, args, timestep='m3'):

        super(CNN1D_LSTM, self).__init__()
        self.cfg = args

        self.device = self.cfg.device



        #self.batch_size = self.args_config.General.batch_size

        if(timestep == 'm3'):

            self.n_xseq = self.cfg.x_frames
            self.n_features = self.cfg.input_dim

            self.hidden_dim = self.cfg.LSTM_m3.hidden_dim
            self.num_layers = self.cfg.LSTM_m3.n_layers

            self.n_kernel_f = self.cfg.CNN1_m3.n_kernel
            self.kernel_size_f = self.cfg.CNN1_m3.kernel_size
            self.n_padding_f = self.cfg.CNN1_m3.n_padding
            self.n_stride_f = self.cfg.CNN1_m3.n_stride

            self.activation_opt_f = self.cfg.CNN1_m3.activation_opt

            self.use_conv_ts = self.cfg.use_second_conv_m3

            if (self.use_conv_ts == True):
                self.n_kernel_ts = self.cfg.CNN2_m3.n_kernel
                self.kernel_size_ts = self.cfg.CNN2_m3.kernel_size
                self.n_padding_ts = self.cfg.CNN2_m3.n_padding
                self.n_stride_ts = self.cfg.CNN2_m3.n_stride
                self.pooling_type = self.cfg.CNN2_m3.pooling_type
                self.n_pool_kernel_size_ts = self.cfg.CNN2_m3.n_pool_kernel_size

                self.activation_opt_ts = self.cfg.CNN2_m3.activation_opt


        elif(timestep == 'm30'):

            self.n_xseq = self.cfg.x_frames_m30
            self.n_features = self.cfg.input_dim_m30

            self.hidden_dim = self.cfg.LSTM_m30.hidden_dim
            self.num_layers = self.cfg.LSTM_m30.n_layers

            self.n_kernel_f = self.cfg.CNN1_m30.n_kernel
            self.kernel_size_f = self.cfg.CNN1_m30.kernel_size
            self.n_padding_f = self.cfg.CNN1_m30.n_padding
            self.n_stride_f = self.cfg.CNN1_m30.n_stride

            self.activation_opt_f = self.cfg.CNN1_m30.activation_opt

            self.use_conv_ts = self.cfg.use_second_conv_m30

            if (self.use_conv_ts == True):
                self.n_kernel_ts = self.cfg.CNN2_m30.n_kernel
                self.kernel_size_ts = self.cfg.CNN2_m30.kernel_size
                self.n_padding_ts = self.cfg.CNN2_m30.n_padding
                self.n_stride_ts = self.cfg.CNN2_m30.n_stride

                self.pooling_type = self.cfg.CNN2_m30.pooling_type
                self.n_pool_kernel_size_ts = self.cfg.CNN2_m30.n_pool_kernel_size
                self.activation_opt_ts = self.cfg.CNN2_m30.activation_opt


        if (self.use_conv_ts == True):
            self.n_cout_seq = int(
                (self.n_xseq + 2 * self.n_padding_ts - (self.kernel_size_ts - 1) - 1) / self.n_stride_ts + 1)

            self.n_cout_seq = int(self.n_cout_seq / self.n_pool_kernel_size_ts)

            self.conv_ts = nn.Conv1d(in_channels=1, out_channels=self.n_kernel_ts,
                                    kernel_size=self.kernel_size_ts, padding=self.n_padding_ts, stride=self.n_stride_ts)

            if self.pooling_type == 'MAX':
                self.pool_ts = nn.MaxPool1d(self.n_pool_kernel_size_ts)
            elif self.pooling_type == 'AVG':
                self.pool_ts = nn.AvgPool1d(self.n_pool_kernel_size_ts)
            else:
                raise Exception("pooling type 설정에 오류가 있습니다. 'MAX' 혹은 'AVG'로 되어있어야 합니다")

        else:
            self.n_cout_seq = self.n_xseq

        # conv_1을 통과 하고 난 이후의 feature의 갯수
        self.n_cout_features = int(
            ((self.n_features + 2 * self.n_padding_f - (self.kernel_size_f - 1) - 1) / self.n_stride_f) + 1)

        self.conv_f = nn.Conv1d(in_channels=1, out_channels=self.n_kernel_f,
                                kernel_size=self.kernel_size_f, padding=self.n_padding_f, stride=self.n_stride_f)


        self.lstm = nn.LSTM(self.n_cout_features, self.hidden_dim, self.num_layers)

        #self.hidden = self.init_hidden()

    """
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    """

    def forward(self, input_data):

        # input data :  [ self.n_xseq,  self.batch_size, self.n_features]
        #x = x.permute(1, 0, 2)  # [ self.batch_size, self.n_xseq, self.n_features]
        x = input_data.reshape(-1, self.n_xseq, self.n_features)

        n_batch = x.shape[0]
        init_hidden = (torch.zeros(self.num_layers, n_batch, self.hidden_dim),
                  torch.zeros(self.num_layers, n_batch, self.hidden_dim))

        hidden = [hidden.to(self.cfg.device) for hidden in init_hidden]

        """ '추출된 features 별'로 '동일한' conv_2 layer 적용하여 연속된 time step pattern 추출 """
        if (self.use_conv_ts == True):
            lst_features = []

            for f in range(self.n_features):
                # c1_out[:, :, f].unsqueeze(1) -->  [ self.batch_size, 1, self.n_xseq ]
                # c_out_per_feature -->  [ self.batch_size, n_kerneal_2, self.n_xseq ]

                if self.activation_opt_ts == 'ReLU':
                    relu = nn.ReLU()
                    c_out_per_feature = relu(self.conv_ts(x[:, :, f].unsqueeze(1)))
                elif self.activation_opt_ts == 'LeakyReLU':
                    leakyReLU = nn.leakyReLU()
                    c_out_per_feature = leakyReLU(self.conv_ts(x[:, :, f].unsqueeze(1)))
                elif self.activation_opt_ts == 'GELU':
                    gelu2 = nn.GELU()
                    c_out_per_feature = gelu2(self.conv_ts(x[:, :, f].unsqueeze(1)))

                elif self.activation_opt_ts == 'TanH':
                    tanh_ts = nn.Tanh()
                    c_out_per_feature = tanh_ts(self.conv_ts(x[:, :, f].unsqueeze(1)))

                else:
                    c_out_per_feature = self.conv_ts(x[:, :, f].unsqueeze(1))

                c_out_per_feature = self.pool_ts(c_out_per_feature)

                # c_out_per_feature -->  [ self.batch_size, n_kerneal_2, n_cout_seq ]
                c_out_per_feature = c_out_per_feature.mean(1, keepdim=False).unsqueeze(2)

                lst_features.append(c_out_per_feature)

            cnn_ts_out = torch.cat(lst_features, axis=2)

            # c_out을 lstm에 넣을 수 있는 형태로 바꾼다.
            cnn1_out = cnn_ts_out.contiguous().view(self.n_cout_seq, n_batch, -1)  # (timesteps, samples, output_size)

        else:
            cnn1_out = x.contiguous().view(self.n_cout_seq, n_batch, -1)


        lst_tstep = []

        """ 'time step 별'로 '동일한' conv_f layer 적용하여 feature 추출 """

        for t in range(self.n_cout_seq):

            # x[:, t, :].unsqueeze(1) ---> [ batch_size, 1, n_features ]
            # cnn_f_out_per_tstep --->  [ batch_size, n_kerneal_1, n_cout_features]

            if self.activation_opt_f == 'ReLU':
                relu = nn.ReLU()
                cnn_f_out_per_tstep = relu(self.conv_f(cnn1_out[t, :, :].unsqueeze(1)))
            elif self.activation_opt_f == 'LeakyReLU':
                leakyReLU = nn.leakyReLU()
                cnn_f_out_per_tstep = leakyReLU(self.conv_f(cnn1_out[t, :, :].unsqueeze(1)))
            elif self.activation_opt_f == 'GeLU':

                gelu1 = nn.GELU()
                cnn_f_out_per_tstep = gelu1(self.conv_f(cnn1_out[t, :, :].unsqueeze(1)))
            else:
                cnn_f_out_per_tstep = self.conv_f(cnn1_out[t,:, :].unsqueeze(1))

            # cnn_f_out_per_tstep --->  [ batch_size, 1, n_cout_features]
            cnn_f_out_per_tstep = cnn_f_out_per_tstep.mean(1, keepdim=False).unsqueeze(1)  # out : [self.batch_size, 1, self.n_cout_features]
            lst_tstep.append(cnn_f_out_per_tstep)

        # c1_out --->  [ batch_size, n_xseq, n_cout_features]
        cnn_f_out = torch.cat(lst_tstep, axis=1)  # c_out = [ self.batch_size, self.n_xseq, self.n_cout_features]

        # c_out을 lstm에 넣을 수 있는 형태로 바꾼다.
        cnn_out = cnn_f_out.contiguous().view(self.n_cout_seq, n_batch, -1)  # (timesteps, samples, output_size)
        #self.hidden = [hidden.to(self.args_config.General.device) for hidden in self.init_hidden()]

        lstm_out, hidden = self.lstm(cnn_out, hidden)

        return lstm_out[-1,:,:]


class MT_CNN_LSTM_B(nn.Module):

    def __init__(self, args_config):
        super(MT_CNN_LSTM_B, self).__init__()

        self.cfg = args_config
        self.n_yseq = self.cfg.y_frames
        self.n_yseq_m30 = self.cfg.y_frames_m30


        self.activation_opt_final = self.cfg.activation_opt_final
        self.activation_opt_final_m30 = self.cfg.LSTM_m30.activation_opt_final

        self.CNN1D_LSTM_m3 = CNN1D_LSTM(self.cfg, 'm3')
        self.CNN1D_LSTM_m30 = CNN1D_LSTM(self.cfg, 'm30')

        self.final_input_dim = self.CNN1D_LSTM_m3.hidden_dim + self.CNN1D_LSTM_m30.hidden_dim
        self.final_input_dim_m30 = self.CNN1D_LSTM_m30.hidden_dim

        self.dropout_val = self.cfg.dropout
        self.dropout_val_m30 = self.cfg.LSTM_m30.dropout
        self.do = nn.Dropout(self.dropout_val)

        self.use_final_bn = self.cfg.use_final_bn
        self.use_final_bn_m30 = self.cfg.LSTM_m30.use_final_bn

        self.bn = nn.BatchNorm1d(self.final_input_dim)
        self.bn_m30 = nn.BatchNorm1d(self.final_input_dim_m30)

        self.final_regressor = self.make_final_regressor()
        self.final_regressor_m30 = self.make_final_regressor_m30()

        self.y_frames_m30 = self.cfg.y_frames_m30

        #self.FC_m30 = nn.Linear(self.CNN1D_LSTM_m30.hidden_dim, self.y_frames_m30)

        if(self.cfg.model_init.method == 'normal_distribution'):
            self.apply(self._init_weights_normal_dist)
        elif (self.cfg.model_init.method == 'xavier'):
            self.apply(self._init_weights_xavier)
        else:
            raise Exception("config의 model_init가 잘못 설정되어있습니다")


    def make_final_regressor(self):
        layers = []

        if self.use_final_bn == True:
            layers.append(nn.BatchNorm1d(self.final_input_dim ))
        layers.append(nn.Dropout(self.dropout_val))

        layers.append(nn.Linear(self.final_input_dim, self.final_input_dim // 2))

        if self.activation_opt_final == 'ReLU':
            layers.append(nn.ReLU())

        elif self.activation_opt_final == 'LeakyReLU':
            layers.append(nn.leakyReLU())

        elif self.activation_opt_final == 'GeLU':
            layers.append(nn.GELU())

        elif self.activation_opt_final == 'TanH':
            layers.append(nn.Tanh())


        layers.append(nn.Linear(self.final_input_dim // 2, self.n_yseq))

        regressor = nn.Sequential(*layers)

        return regressor

    def make_final_regressor_m30(self):
        layers = []

        if self.use_final_bn_m30 == True:
            layers.append(nn.BatchNorm1d(self.final_input_dim_m30 ))
        layers.append(nn.Dropout(self.dropout_val_m30))

        layers.append(nn.Linear(self.final_input_dim_m30, self.final_input_dim_m30 // 2))

        if self.activation_opt_final_m30 == 'ReLU':
            layers.append(nn.ReLU())

        elif self.activation_opt_final_m30 == 'LeakyReLU':
            layers.append(nn.leakyReLU())

        elif self.activation_opt_final_m30 == 'GeLU':
            layers.append(nn.GELU())

        elif self.activation_opt_final_m30 == 'TanH':
            layers.append(nn.Tanh())


        layers.append(nn.Linear(self.final_input_dim_m30 // 2, self.n_yseq_m30))

        regressor = nn.Sequential(*layers)

        return regressor

    def _init_weights_normal_dist(self, module):
        torch.manual_seed(self.cfg.seed)
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.BatchNorm1d):
            module.weight.data.normal_(mean=0.0, std=self.cfg.model_init.range_val)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=self.cfg.model_init.range_val)


    def _init_weights_xavier(self, module):
        torch.manual_seed(self.cfg.seed)
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, nn.BatchNorm1d):
            module.weight.data.normal_(mean=0.0, std=self.cfg.model_init.range_val)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self,x_m3, x_m30):

        out_m3 = self.CNN1D_LSTM_m3(x_m3)
        out_m30 = self.CNN1D_LSTM_m30(x_m30)

        tot_lstm_out = torch.cat([out_m3, out_m30.detach()], dim=1)
        #tot_lstm_out = torch.cat([out_m3, out_m30], dim=1)
        final_out = self.final_regressor(tot_lstm_out)
        final_out_m30 = self.final_regressor_m30(out_m30)

        return final_out, final_out_m30

    def predict_m30(self, x_m30):

        out_m30 = self.CNN1D_LSTM_m30(x_m30)
        final_out_m30 = self.final_regressor_m30(out_m30)

        return final_out_m30









