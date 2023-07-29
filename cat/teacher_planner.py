import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet18
from copy import deepcopy

class TeacherPlanner(nn.Module):
    def __init__(self,
        pixels_per_meter=2, crop_size=64, x_offset=0, y_offset=0.75,
        feature_x_jitter=1, feature_angle_jitter=10, 
        num_plan=10, k=16, num_out_feature=64, num_cmds=6, num_plan_iter=1,
        ):

        super().__init__()
        
        self.num_cmds = num_cmds
        self.num_plan = num_plan
        self.num_plan_iter = num_plan_iter

        self.num_out_feature = num_out_feature

        self.pixels_per_meter = pixels_per_meter
        self.crop_size = crop_size

        self.feature_x_jitter = feature_x_jitter
        self.feature_angle_jitter = np.deg2rad(feature_angle_jitter)

        self.offset_x = nn.Parameter(torch.tensor(x_offset).float(), requires_grad=False)
        self.offset_y = nn.Parameter(torch.tensor(y_offset).float(), requires_grad=False)

        self.bev_conv_emb = nn.Sequential(
            resnet18(num_channels=21),
            # resnet34(num_channels=8),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        self.plan_gru = nn.GRU(4,640,batch_first=True)
        self.plan_mlp = nn.Linear(640,2)

        self.cast_grus = nn.ModuleList([nn.GRU(640, 64, batch_first=True) for _ in range(self.num_cmds)])
        self.cast_mlps = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.num_cmds)])
        self.cast_cmd_pred = nn.Sequential(
            nn.Linear(640,self.num_cmds),
            nn.Sigmoid(),
        )

    def infer(self, bev, nxps, cmd, spd):

        B = bev.size(0)
        assert bev.shape == torch.Size([B, 21, 192, 192])

        cropped_ego_bev = bev

        assert cropped_ego_bev.shape == torch.Size([B, 21, 192, 192])

        ego_bev_embd = self.bev_conv_emb(cropped_ego_bev)

        spd = spd[...,None].repeat((1,128))

        feat_cat = torch.cat((ego_bev_embd, spd), dim=1)

        ego_cast_locs = self.cast(feat_cat)

        ego_plan_locs = self.plan(
            feat_cat, nxps, 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )[0,-1,cmd]

        ego_cast_cmds = self.cast_cmd_pred(feat_cat)

        return ego_plan_locs, ego_cast_locs[0,cmd], ego_cast_cmds



    def forward(self, bev, nxps, spd):

        B = bev.size(0)

        assert bev.shape == torch.Size([B, 21, 192, 192])

        cropped_ego_bev = bev

        assert cropped_ego_bev.shape == torch.Size([B, 21, 192, 192])

        ego_bev_embd = self.bev_conv_emb(cropped_ego_bev)

        spd = spd[...,None].repeat((1,128))

        feat_cat = torch.cat((ego_bev_embd, spd), dim=1)


        ego_cast_locs = self.cast(feat_cat)

        ego_plan_locs = self.plan(
            feat_cat, nxps, 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )

        ego_cast_cmds = self.cast_cmd_pred(feat_cat)

        return (
            ego_plan_locs, ego_cast_locs, ego_cast_cmds
        )


    def _plan(self, embd, nxp, cast_locs, pixels_per_meter=4, crop_size=96):

        B = embd.size(0)

        h0, u0 = embd, nxp*pixels_per_meter/crop_size*2-1

        self.plan_gru.flatten_parameters()

        locs = []
        for i in range(self.num_cmds):
            u = torch.cat([
                u0.expand(self.num_plan, B, -1).permute(1,0,2),
                cast_locs[:,i]
            ], dim=2)
            out, _ = self.plan_gru(u, h0[None])
            locs.append(torch.cumsum(self.plan_mlp(out), dim=1))

        return torch.stack(locs, dim=1) + cast_locs
    
    def plan(self, embd, nxp, cast_locs=None, pixels_per_meter=4, crop_size=96):

        if cast_locs is None:
            plan_loc = self.cast(embd).detach()
        else:
            plan_loc = cast_locs.detach()
        
        plan_locs = []
        for i in range(self.num_plan_iter):
            plan_loc = self._plan(embd, nxp, plan_loc, pixels_per_meter=pixels_per_meter, crop_size=crop_size)
            plan_locs.append(plan_loc)

        return torch.stack(plan_locs, dim=1)

    def cast(self, embd):
        B = embd.size(0)

        u = embd.expand(self.num_plan, B, -1).permute(1,0,2)

        locs = []
        for gru, mlp in zip(self.cast_grus, self.cast_mlps):
            gru.flatten_parameters()
            out, _ = gru(u)
            locs.append(torch.cumsum(mlp(out), dim=1))

        return torch.stack(locs, dim=1)

    

def transform_points(locs, oris):
    cos, sin = torch.cos(oris), torch.sin(oris)
    R = torch.stack([
        torch.stack([ cos, sin], dim=-1),
        torch.stack([-sin, cos], dim=-1),
    ], dim=-2)

    return locs @ R


def filter_cars(ego_locs, locs, typs):
    # We don't care about cars behind us ;)
    rel_locs = locs[:,:,0] - ego_locs[:,0:1]

    return typs & (rel_locs[...,1] < 0)


def random_sample(binaries, size):

    cut_binaries = torch.zeros_like(binaries)
    for i in range(binaries.size(0)):
        if binaries[i].sum() <= size:
            cut_binaries[i] = binaries[i]
        else:
            nonzero = torch.nonzero(binaries[i]).squeeze(1)
            nonzero_idx = torch.multinomial(torch.ones_like(nonzero).float(), size)
            nonzero = nonzero[nonzero_idx]
            cut_binaries[i,nonzero] = binaries[i,nonzero]
    
    return cut_binaries