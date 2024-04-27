import matplotlib.pyplot as plt
import os
import torch
from torch import nn, einsum
from einops import rearrange
class Aggregate(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out

class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))

        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)
        self.relu = nn.ReLU()
    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        q=self.relu(q)
        k=self.relu(k)

        sim = einsum('b h x y d, b h u v d -> b h x y u v',q , k)


        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        max= sim.max(dim=-1,keepdims=True)[0]
        sim = sim - max
        attn = sim.softmax(dim=-1)

        att = rearrange(attn, ' b h (x y) (u v) -> b h x y u v',x=h,y=w,u=h,v=w)

        # coordinates = [(116,16)]
        # for i, (x,y) in enumerate(coordinates):
        #     att_map = att[0,3,y,x].detach().cpu().numpy()
        #     plt.imshow(att_map,cmap='hot')
        #    # plt.text(x-8,y+1,f'{i+1}',color='w',size=15)
        #     plt.scatter(x,y, s=120,color='w',marker='o')
        #     plt.axis= 'off'
        #     frame = plt.gca()
        #
        #     frame.axes.get_yaxis().set_visible(False)
        #
        #     frame.axes.get_xaxis().set_visible(False)
        #     plt.savefig(os.path.join('./',f'x_{i+1}_{x}_y_{y}_att_map.png'),bbox_inches='tight',pad_inches=-0.10)
        #     plt.close()
        return attn


class Aggregate(nn.Module):
    def __init__(
        self,

        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1)) #-0.128

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


if __name__ == "__main__":
    att = Attention(dim=128, heads=1)
    fmap = torch.randn(2, 128, 40, 90)
    out = att(fmap)

    print(out.shape)
