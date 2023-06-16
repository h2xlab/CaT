import torch
import torch.nn as nn
from cat.utils.utils import *

from cat.models.networks.feature_extractor import *
from cat.models.networks import Lane2D, Lane3D
from cat.models.networks.libs.layers import *
from cat.models.networks.PE import PositionEmbeddingLearned
from cat.models.networks.Layers import EncoderLayer
from cat.models.networks.Unet_parts import Down, Up
from cat.models.networks.fpn import FPN

class Normalize(nn.Module):
    """ ImageNet normalization """
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, device=device), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std, device=device), requires_grad=False)

    def forward(self, x):
        return (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]


# overall network
class CatPlanner(nn.Module):
    def __init__(self, batch_size, num_proj, encoder, feature_channels, nhead, npoints, 
                crop_size=96, pixels_per_meter=4, num_plan=10, num_cmds=6, num_plan_iter=1, num_classes=9):
        super(CatPlanner, self).__init__()

        self.batch_size = batch_size

        self.num_proj = num_proj

        # define required transformation matrices
        top_view_region = np.array([[38.4, -19.2], [38.4, 19.2], [0, -19.2], [0, 19.2]]) # carla xy is different

        bev_h, bev_w = crop_size*2, crop_size*2
        cam_xyz = [-1.5, 0, 2.0] #[1.3, 0, 2.3]
        cam_yaws = [-60, 0, 60]

        width = 480
        height = 224
        fov = 64

        top_crop = 46
        
        focal = width / (2.0 * np.tan(fov * np.pi / 360.0)) 
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = width / 2.0
        K[1, 2] = (height+top_crop) / 2.0

        K[1, 2] -= top_crop


        self.M_inv_left = homography_im2ipm_norm_jim(top_view_region, width, height, cam_xyz, cam_yaw=-60, K=K)#.cuda()
        self.M_inv_middle = homography_im2ipm_norm_jim(top_view_region, width, height, cam_xyz, cam_yaw=0, K=K)#.cuda()
        self.M_inv_right = homography_im2ipm_norm_jim(top_view_region, width, height, cam_xyz, cam_yaw=60, K=K)#.cuda()


        # Define network
        # backbone: feature_extractor
        self.encoder = self.get_encoder(encoder)
        self.fpn = FPN(self.encoder.dimList, feature_channels, 4, add_extra_convs=False)
        self.neck = nn.Sequential(*make_one_layer(self.encoder.dimList[0], feature_channels, batch_norm=True),
                                  *make_one_layer(feature_channels, feature_channels, batch_norm=True))

        # 2d detector
        self.shared_encoder = Lane2D.FrontViewPathway(feature_channels, num_proj)


        '''
            ATTENTION RELATED
            frontview_features_0 size: torch.Size([4, 128, 180, 240])
            frontview_features_1 size: torch.Size([4, 256, 90, 120])
            frontview_features_2 size: torch.Size([4, 512, 45, 60])
            frontview_features_3 size: torch.Size([4, 512, 22, 30])
            x_0 size: torch.Size([4, 128, 208, 128])
            x_1 size: torch.Size([4, 128, 104, 64])
            x_2 size: torch.Size([4, 256, 52, 32])
            x_3 size: torch.Size([4, 256, 26, 16])
        '''
        # attn num channel
        # self.uv_feat_c_1 = 128
        self.uv_feat_c_1 = 64
        self.uv_feat_c_2 = self.uv_feat_c_1 * 2
        self.uv_feat_c_3 = self.uv_feat_c_2 * 2

        self.uv_h_1 = 56
        self.uv_w_1 = 120 * 3
        self.uv_feat_len_1 = self.uv_h_1 * self.uv_w_1

        self.uv_h_2 = self.uv_h_1 // 2
        self.uv_w_2 = self.uv_w_1 // 2
        self.uv_feat_len_2 = self.uv_h_2 * self.uv_w_2

        self.uv_h_3 = self.uv_h_2 // 2
        self.uv_w_3 = self.uv_w_2 // 2
        self.uv_feat_len_3 = self.uv_h_3 * self.uv_w_3

        self.bev_h_1 = (crop_size*2) // 2
        self.bev_w_1 = (crop_size*2) // 2
        self.bev_feat_len_1 = self.bev_h_1 * self.bev_w_1
        
        self.bev_h_2 = self.bev_h_1 // 2
        self.bev_w_2 = self.bev_w_1 // 2
        self.bev_feat_len_2 = self.bev_h_2 * self.bev_w_2

        self.bev_h_3 = self.bev_h_2 // 2
        self.bev_w_3 = self.bev_w_2 // 2
        self.bev_feat_len_3 = self.bev_h_3 * self.bev_w_3

        self.dim_ffn_3 = self.uv_feat_c_3 * 2
        self.dim_ffn_2 = self.uv_feat_c_2 * 2
        self.dim_ffn_1 = self.uv_feat_c_1 * 2

        self.nhead = nhead

        # learnable query
        query_embed_1 = nn.Embedding(self.bev_feat_len_1, self.uv_feat_c_1)
        query_embed_2 = nn.Embedding(self.bev_feat_len_2, self.uv_feat_c_2)
        query_embed_3 = nn.Embedding(self.bev_feat_len_3, self.uv_feat_c_3)
        # query_embed_4 = nn.Embedding(self.bev_feat_len_4, self.uv_feat_c_4)

        self.query_embeds = nn.ModuleList()
        self.query_embeds.append(query_embed_1)
        self.query_embeds.append(query_embed_2)
        self.query_embeds.append(query_embed_3)
        # self.query_embeds.append(query_embed_4)

        self.npoints = npoints

        # Encoder layer version
        el1 = EncoderLayer(d_model=self.uv_feat_c_1, dim_ff=self.uv_feat_c_1*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el2 = EncoderLayer(d_model=self.uv_feat_c_2, dim_ff=self.uv_feat_c_2*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)
        el3 = EncoderLayer(d_model=self.uv_feat_c_3, dim_ff=self.uv_feat_c_3*2, num_levels=1, num_points=self.npoints, num_heads=self.nhead)

        self.el = nn.ModuleList()
        self.el.append(el1)
        self.el.append(el2)
        self.el.append(el3)

        pe1 = PositionEmbeddingLearned(h=self.bev_h_1, w=self.bev_w_1, num_pos_feats=self.uv_feat_c_1 // 2)
        pe2 = PositionEmbeddingLearned(h=self.bev_h_2, w=self.bev_w_2, num_pos_feats=self.uv_feat_c_2 // 2)
        pe3 = PositionEmbeddingLearned(h=self.bev_h_3, w=self.bev_w_3, num_pos_feats=self.uv_feat_c_3 // 2)
        self.pe = nn.ModuleList()
        self.pe.append(pe1)
        self.pe.append(pe2)
        self.pe.append(pe3)

        # 2d uniform sampling
        self.ref_2d_1 = self.get_reference_points(H=self.bev_h_1, W=self.bev_w_1, dim='2d', bs=1)
        self.ref_2d_2 = self.get_reference_points(H=self.bev_h_2, W=self.bev_w_2, dim='2d', bs=1)
        self.ref_2d_3 = self.get_reference_points(H=self.bev_h_3, W=self.bev_w_3, dim='2d', bs=1)

        size_top1 = torch.Size([self.bev_h_1, self.bev_w_1])
        self.project_layer1 = Lane3D.RefPntsNoGradGenerator(size_top1, self.M_inv_right, self.M_inv_middle, self.M_inv_left, False)

        size_top2 = torch.Size([self.bev_h_2, self.bev_w_2])
        self.project_layer2 = Lane3D.RefPntsNoGradGenerator(size_top2, self.M_inv_right, self.M_inv_middle, self.M_inv_left, False)

        size_top3 = torch.Size([self.bev_h_3, self.bev_w_3])
        self.project_layer3 = Lane3D.RefPntsNoGradGenerator(size_top3, self.M_inv_right, self.M_inv_middle, self.M_inv_left, False)

        # input_spatial_shapes & input_level_start_index
        self.input_spatial_shapes_1 = torch.as_tensor([(self.uv_h_1, self.uv_w_1)], dtype=torch.long)
        self.input_level_start_index_1 = torch.as_tensor([0.0,], dtype=torch.long)

        self.input_spatial_shapes_2 = torch.as_tensor([(self.uv_h_2, self.uv_w_2)], dtype=torch.long)
        self.input_level_start_index_2 = torch.as_tensor([0.0,], dtype=torch.long)

        self.input_spatial_shapes_3 = torch.as_tensor([(self.uv_h_3, self.uv_w_3)], dtype=torch.long)
        self.input_level_start_index_3 = torch.as_tensor([0.0,], dtype=torch.long)

        self.batch_norm = True

        self.dim_size_rts_1 = Lane3D.SingleTopViewPathway(64)

        self.dim_rts_1 = nn.Sequential(*make_one_layer(128,
                                                    64,
                                                    kernel_size=1,
                                                    padding=0,
                                                    batch_norm=self.batch_norm))


        self.dim_size_rts_2 = Lane3D.SingleTopViewPathway(128)

        self.dim_rts_2 = nn.Sequential(*make_one_layer(256,
                                                    128,
                                                    kernel_size=1,
                                                    padding=0,
                                                    batch_norm=self.batch_norm))

        self.dim_size_rts_3 = Lane3D.SingleTopViewPathway(256)

        self.dim_size_rts_4 = Lane3D.SingleUpTopViewPathway(256)

        # single att-style ipm
        self.use_proj_layer = 0

        '''
            projs_0 size: torch.Size([4, 128, 208, 128])
            projs_1 size: torch.Size([4, 256, 104, 64])
            projs_2 size: torch.Size([4, 512, 52, 32])
            projs_3 size: torch.Size([4, 512, 26, 16])
        '''

        self.bev_conv_emb = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        self.pixels_per_meter = pixels_per_meter
        self.crop_size = crop_size

        self.num_cmds = num_cmds
        self.num_plan = num_plan
        self.num_plan_iter = num_plan_iter

        self.plan_gru = nn.GRU(4,512,batch_first=True)
        self.plan_mlp = nn.Linear(512,2)

        self.cast_grus = nn.ModuleList([nn.GRU(512, 64, batch_first=True) for _ in range(self.num_cmds)])
        self.cast_mlps = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.num_cmds)])
        self.cast_cmd_pred = nn.Sequential(
            nn.Linear(512,self.num_cmds),
            nn.Sigmoid(),
        )

        self.down1 = Down(64, 256)
        self.down2 = Down(256, 512)
        factor = 2
        self.down3 = Down(512, 1024//factor)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(192, 128)

        # segmentation head
        self.segment_head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, num_classes, kernel_size=1),
                )

    def forward(self, img, nxps, use_fpn=False):
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=img.device)
        out_featList = self.encoder(self.normalize(img/255.))
        frontview_features = out_featList[1:]

        assert len(frontview_features) == 3

        projs = []
        # deform att multi scale
        for i in range(3):
            input_spatial_shapes = getattr(self, "input_spatial_shapes_{}".format(i + 1)).to(img.device)
            input_level_start_index = getattr(self, "input_level_start_index_{}".format(i + 1)).to(img.device)
            bs, c, _, _ = frontview_features[i].shape

            bev_h = getattr(self, "bev_h_{}".format(i + 1))
            bev_w = getattr(self, "bev_w_{}".format(i + 1))

            src = frontview_features[i].flatten(2).permute(0, 2, 1)
            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)

            # reference points generated by ipm grid
            project_layer = getattr(self, "project_layer{}".format(i + 1))

            ref_pnts = project_layer(bs, img.device).unsqueeze(-2)

            # encoder layers
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(query_embed.dtype)
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)

            ref_2d = getattr(self, 'ref_2d_{}'.format(i + 1)).repeat(bs, 1, 1, 1).to(img.device)

            x = self.el[i](query=query_embed, value=src, bev_pos=bev_pos, 
                                        ref_2d = ref_2d, ref_3d=ref_pnts,
                                        bev_h=bev_h, bev_w=bev_w, 
                                        spatial_shapes=input_spatial_shapes,
                                        level_start_index=input_level_start_index)

            x = x.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()

            projs.append(x)

        down_feat_1 = self.dim_size_rts_1(projs[0]) # 48x48x64

        proj1_feat = self.dim_rts_1(projs[1])
        cat_feat_1 = torch.cat((down_feat_1, proj1_feat), 1)

        down_feat_2 = self.dim_size_rts_2(cat_feat_1) # 24x24x128

        proj2_feat = self.dim_rts_2(projs[2])
        cat_feat_2 = torch.cat((down_feat_2, proj2_feat), 1)

        down_feat_3 = self.dim_size_rts_3(cat_feat_2) # 12x12x256

        _x = self.dim_size_rts_4(down_feat_3) # 6x6x512

        ego_bev_embd = self.bev_conv_emb(_x)

        ego_cast_locs = self.cast(ego_bev_embd)

        ego_plan_locs = self.plan(
            ego_bev_embd, nxps, 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )

        ego_cast_cmds = self.cast_cmd_pred(ego_bev_embd)

        x1 = self.down1(projs[0])
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x_out = self.up1(x3, x2)
        x_out = self.up2(x_out, x1)
        x_out = self.up3(x_out, projs[0])

        pred_seg_bev_map = self.segment_head(x_out)

        return (ego_plan_locs, ego_cast_locs, ego_cast_cmds), pred_seg_bev_map



    def infer(self, img, nxps, cmd):
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=img.device)

        out_featList = self.encoder(self.normalize(img/255.))
        frontview_features = out_featList[1:]

        assert len(frontview_features) == 3

        projs = []

        for i in range(3):
            input_spatial_shapes = getattr(self, "input_spatial_shapes_{}".format(i + 1)).to(img.device)
            input_level_start_index = getattr(self, "input_level_start_index_{}".format(i + 1)).to(img.device)
            bs, c, _, _ = frontview_features[i].shape

            bev_h = getattr(self, "bev_h_{}".format(i + 1))
            bev_w = getattr(self, "bev_w_{}".format(i + 1))

            src = frontview_features[i].flatten(2).permute(0, 2, 1)

            query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)

            # reference points generated by ipm grid
            project_layer = getattr(self, "project_layer{}".format(i + 1))

            ref_pnts = project_layer(bs, img.device).unsqueeze(-2)

            # encoder layers
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=query_embed.device).to(query_embed.dtype)
            bev_pos = self.pe[i](bev_mask).to(query_embed.dtype)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)


            ref_2d = getattr(self, 'ref_2d_{}'.format(i + 1)).repeat(bs, 1, 1, 1).to(img.device)

            x = self.el[i](query=query_embed, value=src, bev_pos=bev_pos, 
                                        ref_2d = ref_2d, ref_3d=ref_pnts,
                                        bev_h=bev_h, bev_w=bev_w, 
                                        spatial_shapes=input_spatial_shapes,
                                        level_start_index=input_level_start_index)

            x = x.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()

            projs.append(x)

        down_feat_1 = self.dim_size_rts_1(projs[0]) # 48x48x64
        proj1_feat = self.dim_rts_1(projs[1])
        cat_feat_1 = torch.cat((down_feat_1, proj1_feat), 1)
        down_feat_2 = self.dim_size_rts_2(cat_feat_1) # 24x24x128
        proj2_feat = self.dim_rts_2(projs[2])
        cat_feat_2 = torch.cat((down_feat_2, proj2_feat), 1)
        down_feat_3 = self.dim_size_rts_3(cat_feat_2) # 12x12x256
        _x = self.dim_size_rts_4(down_feat_3) # 6x6x512


        ego_bev_embd = self.bev_conv_emb(_x)

        ego_cast_locs = self.cast(ego_bev_embd)

        ego_plan_locs = self.plan(
            ego_bev_embd, nxps, 
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter, 
            crop_size=self.crop_size*2
        )[0,-1,cmd]

        ego_cast_cmds = self.cast_cmd_pred(ego_bev_embd)

        # segment head v3
        x1 = self.down1(projs[0])
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x_out = self.up1(x3, x2)
        x_out = self.up2(x_out, x1)
        x_out = self.up3(x_out, projs[0])

        pred_seg_bev_map = self.segment_head(x_out)

        return ego_plan_locs, ego_cast_locs[0,cmd], ego_cast_cmds, pred_seg_bev_map



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









    @staticmethod
    def get_reference_points(H, W, Z=8, D=4, dim='3d', bs=1, device='cuda', dtype=torch.long):
        """Get the reference points used in decoder.
        Args:
            H, W spatial shape of bev
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # 2d to 3d reference points, need grid from M_inv
        if dim == '3d':
            raise Exception("get reference poitns 3d not supported")
            zs = torch.linspace(0.5, Z - 0.5, D, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(-1, H, W) / Z

            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(D, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(D, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H  # ?
            ref_x = ref_x.reshape(-1)[None] / W  # ?
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d   



    def get_encoder(self, encoder):
        if encoder == 'ResNext101':
            return deepFeatureExtractor_ResNext101(lv6=False)
        
        elif encoder == 'ResNet34':
            return deepFeatureExtractor_ResNet34(lv6=False)


        elif encoder == 'VGG19':
            return deepFeatureExtractor_VGG19(lv6=False)
        elif encoder == 'DenseNet161':
            return deepFeatureExtractor_DenseNet161(lv6=False)
        elif encoder == 'InceptionV3':
            return deepFeatureExtractor_InceptionV3(lv6=False)
        elif encoder == 'MobileNetV2':
            return deepFeatureExtractor_MobileNetV2(lv6=False)
        elif encoder == 'ResNet101':
            return deepFeatureExtractor_ResNet101(lv6=False)
        elif 'EfficientNet' in encoder:
            return deepFeatureExtractor_EfficientNet(encoder, lv6=False, lv5=True, lv4=True, lv3=True)
        else:
            raise Exception("encoder model in args is not supported")
