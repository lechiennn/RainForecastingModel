import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    
    def __init__(
        self,
        input_channel: int,
        hidden_channel: int,
        kernel_size: (int, int),
        bias: bool
    ) -> None:
        """
        Init ConvLSTM cell .
        Parameters
        ----------
        input_channel: int
            Number of channels of input tensor.
        hidden_channel: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channel + self.hidden_channel,
                              out_channels = 4 * self.hidden_channel,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=bias)
        
        self.conv_c = nn.Conv2d(in_channels=self.hidden_channel,
                                out_channels=2 * self.hidden_channel,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=bias)
        
        self.conv_c_next = nn.Conv2d(in_channels=self.hidden_channel,
                                out_channels=self.hidden_channel,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=bias)
        
    def forward(self, input_tensor, prev_state):
        h_prev, c_prev = prev_state

        combined = torch.cat([input_tensor, h_prev], dim=1)
        combined_conv = self.conv(combined)
        c_prev_conv = self.conv_c(c_prev)

        c_prev_i, c_prev_f = torch.split(c_prev_conv, self.hidden_channel, dim=1)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channel, dim=1)

        i = torch.sigmoid(cc_i + c_prev_i)
        f = torch.sigmoid(cc_f + c_prev_f)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        c_cur_conv = self.conv_c_next(c_cur)
        o = torch.sigmoid(cc_o + c_cur_conv)

        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, image_size):
        heigh, width = image_size
        return (torch.zeros(batch_size, self.hidden_channel, heigh, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channel, heigh, width, device=self.conv.weight.device))

if __name__ == "__main__":
    _ = ConvLSTMCell()
