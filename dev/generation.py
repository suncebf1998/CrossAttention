#:TODO when small shape cannt use conv1d(kernel_size)
import torch
from torch import nn
import einops

class GenerConv1d(nn.Module):

    def __init__(
        self, input_dim, output_dim, inter_ratio=[2., 128, -1.], kernel_size=3, 
        *, endwithact=False):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential()
        old_dim = input_dim
        for ratio in inter_ratio:
            if isinstance(ratio, int) and ratio>0:
                dim = ratio
            else:
                dim = int(ratio * input_dim if ratio >0 else -1 * ratio * output_dim)
            self.model.append(
                nn.Sequential(
                    nn.Conv1d(old_dim, dim, kernel_size=kernel_size),
                    nn.BatchNorm1d(dim),
                    nn.LeakyReLU()
                )
            )
            old_dim = dim
        self.model.append(
            nn.Sequential(
                nn.Conv1d(old_dim, output_dim, kernel_size=kernel_size),
                nn.BatchNorm1d(dim),
                nn.Tanh() if endwithact else nn.Identity()
            )
        )

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == self.input_dim,\
            f"inputs random noise must be in (Batch, Number, {self.input_dim})!."
        return self.model(x)




class GenerationModel_Linear(nn.Module):
    """
    (n, dim) -> (n, m, dim')
    (n, dim) -Linear-> (n, m0 * dim0) -> (n, m0, dim0)
    (n, mi, dimi) -conv-> (n, mi, dimi+1 * ratio) -> (n, mi*ratio->mi+1, dimi+1) * repeat
    eg:
    b, 128 -> b, 1, 512 (0)-> b, 4, 256 (1)-> b, 16, 128 (2)-> b, 64, 64 (3)-> b, 256, 32 (4)
    """
    def __init__(self, input_dim=128, head_shape=(1,512), ratio=2, ratio_shape=4, depth=3, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.num0 = int(head_shape[0])
        self.dim0 = int(head_shape[1])
        self.head_dim = self.num0 * self.dim0 
        self.ratio_shape = ratio_shape
        self.head_gen = nn.Sequential(
            nn.Linear(input_dim, self.head_dim),
            nn.Tanh()
        )
        self.body_gens = nn.ModuleList()
        for i in range(depth):
            indim = int(self.dim0 * (ratio/ratio_shape) ** i)
            dim = indim * ratio
            num = self.num0 * ratio ** (i+1)
            self.body_gens.append(GenerConv1d(
                indim, dim, 
                [1.,2.,-1.],
                1 if num < 7 else kernel_size, 
                endwithact=True if i<depth-1 else False))
    
    def forward(self, x):
        assert x.shape[-1] == self.input_dim and len(x.shape) == 2,\
            f"inputs random noise must be in (Batch, {self.input_dim})!."
        x:torch.Tensor = self.head_gen(x)
        x = einops.rearrange(x, "b (dim num) -> b dim num", num=self.num0) # odim: old dim
        outputs = [x, ]
        for body_gen_model in self.body_gens:
            x = body_gen_model(x)
            x = einops.rearrange(x, "b (ratio dim) num -> b dim (ratio num)", ratio=self.ratio_shape)
            outputs.append(x)
        
        return outputs


if __name__ == "__main__":
    from tqdm import tqdm
    from tools import get_parameter_number
    testx = torch.randn(512, 128).cuda()
    testy = (testx + testx ** 2 + 3.14).mean(1)
    def metric(outputs:list):
        loss = 0
        for output in outputs:
            loss += output.mean((1,2))
        return loss / len(outputs)
    model = GenerationModel_Linear().cuda()
    get_parameter_number(model, True)

    # test whether can run the interfence
    outputs = model(testx)
    for output in outputs:
        print(output.shape)

    # test whether can use in backward
    optim = torch.optim.Adam(model.parameters(),1e-3)
    with tqdm(range(100), desc=f"Train", leave=False) as epoch_iter:
        for i in epoch_iter:
            model.train()
            outputs = model(testx)
            l = ((metric(outputs) - testy) ** 2).mean()
            l.backward()
            optim.step()
            optim.zero_grad()
            epoch_iter.set_postfix(loss=l.item())
    print(l.item())

        




    
   
