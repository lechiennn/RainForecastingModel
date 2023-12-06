import torch
from torch import nn
from conv_lstm_cell import ConvLSTMCell

class ConvLSTM(nn.Module):

    def __init__(
            self,
            input_channel: int,
            hidden_channel: int,
            kernel_size: (int, int),
            num_layers: int,
            batch_first: bool,
            bias: bool,
            return_all_layers: bool
    ) -> None:
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []

        for i in range(0, self.num_layers):
            cur_input_channel = self.input_channel if i == 0 else self.hidden_channel
            cell_list.append(ConvLSTMCell(input_channel=cur_input_channel,
                                          hidden_channel=hidden_channel,
                                          kernel_size=kernel_size,
                                          bias=bias))
            
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):
        '''
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        sequence_length = input_tensor.size(1)
        cur_layer_input = input_tensor

        layer_output_list = []
        last_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(sequence_length):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 prev_state=[h,c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list


    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')


if __name__ == '__main__':
    convLSTM = ConvLSTM(input_channel=1, hidden_channel=1, bias=True,
                        kernel_size=(3,3),num_layers=1,batch_first=True, return_all_layers=True)
    
    input = torch.rand((3,5,1,90,250))
    output = convLSTM(input)
    print(output[0][0].shape) ### stack output cua tat ca shell cua tat ca layer #dim = (return,layer)
    print(output[1][0][0].shape) ###### output here (dim = (return,layer,h))
    print(output[1][0][1].shape) #### c cua last output (dim = (return,layer,h, c))
