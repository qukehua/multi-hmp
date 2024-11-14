import math
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from functools import partial
from model.SRGCN   import GCN


# def generate_mask(joint_num):
# 	mask = torch.zeros(joint_num, joint_num)
# 	mask.fill_diagonal_(float('-inf'))
#
# 	return mask
class PositionalEmbedding(nn.Module):
	def __init__(self, N, J, embed_size, dropout=0.1):
		super().__init__()
		self.N = N
		self.J = J
		self.joint = nn.Parameter(torch.zeros(J, embed_size))
		self.person = nn.Parameter(torch.zeros(N, embed_size))
		self.person_solo = nn.Parameter(torch.zeros(1, embed_size))
		self.dropout = nn.Dropout(p=dropout)
		torch.nn.init.normal_(self.joint, std=.02)
		torch.nn.init.normal_(self.person, std=.02)

	def forward_spatial(self):
		p_person = self.person.repeat_interleave(self.J, dim=0)
		p_joint = self.joint.repeat(self.N, 1)
		p = p_person + p_joint
		return self.dropout(p)

	def forward_external(self):
		p_person = self.person_solo.repeat_interleave(self.J, dim=0)
		#print(p_person.shape)
		p_joint = self.joint.repeat(1, 1)
		#print(p_joint.shape)
		p = p_person + p_joint
		return self.dropout(p)
	
	def forward_relation(self):
		p = self.forward_spatial()
		p_i = p.unsqueeze(-2)
		p_j = p.unsqueeze(-3)
		p = p_i + p_j
		return self.dropout(p)	

def gen_velocity(m):
    input = [m[:, 0, :] - m[:, 0, :]]  # 差值[16  66]

    for k in range(m.shape[1] - 1):
        input.append(m[:, k + 1, :] - m[:, k, :])
    input = torch.stack((input)).permute(1, 0, 2)  # [16 35 66]

    return input
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1024, output_joints=13, output_dim=3):
        """
        参数:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_joints: 预测的关键点数量(例如COCO数据集17个关键点)
        output_dim: 每个关键点的维度(3D坐标xyz为3)
        """
        super(Decoder, self).__init__()
        
        self.mlp = nn.Sequential(
            
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
           
            nn.Linear(hidden_dim // 2, output_joints * output_dim)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x)
        predict = x.view(batch_size, -1, 3)
        return predict

def delta_2_gt(prediction, last_timestep):
    prediction = prediction.clone()

    # print (prediction [:,0,:].shape,last_timestep.shape)
    prediction[:, 0, :] = prediction[:, 0, :] + last_timestep
    for i in range(prediction.shape[1] - 1):
        prediction[:, i + 1, :] = prediction[:, i + 1, :] + prediction[:, i, :]

    return prediction

class MLP(nn.Module):
	def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))


		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)

		return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class InterPersonAttention(nn.Module):
	def __init__(self, d_model, nhead):
		super(InterPersonAttention, self).__init__()
		self.d_model = d_model
		self.nhead = nhead
		self.d_k = d_model // nhead

		self.query = nn.Linear(d_model, d_model)
		self.key = nn.Linear(d_model, d_model)
		self.value = nn.Linear(d_model, d_model)
		self.fc_out = nn.Linear(d_model, d_model)

	def forward(self, x1, x2):
		B, N, _ = x1.size()

		Q = self.query(x1).view(B, N, self.nhead, self.d_k).transpose(1, 2) # shape: (B, nhead, N, d_k)
		K = self.key(x2).view(B, N, self.nhead, self.d_k).transpose(1, 2) # key and value are from x2
		V = self.value(x2).view(B, N, self.nhead, self.d_k).transpose(1, 2)

		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # shape: (B, nhead, N, N)

		attn_probs = F.softmax(attn_scores, dim=-1)

		attn_output = torch.matmul(attn_probs, V) # shape: (B, nhead, N, d_k)
		attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1) # shape: (B, N, d_model)

		output = self.fc_out(attn_output)

		return output








class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.J_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.R_conv = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.R_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, joint_feature, relation_feature, mask=None):
        B, N, C = joint_feature.shape
        H = self.num_heads
        HS = C // self.num_heads
        J_qkv = self.J_qkv(joint_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4) #[3, B, #heads, N, C//#heads]
        R_qkv = self.R_qk(relation_feature).reshape(B, N, N, 2, H, HS).permute(3, 0, 4, 1, 2, 5)  #[3, B, #heads, N, N, C//#heads]

        J_q, J_k, J_v = J_qkv[0], J_qkv[1], J_qkv[2]   #[B, #heads, N, C//#heads]
        R_q, R_k = R_qkv[0], R_qkv[1]  #[B, #heads, N, N, C//#heads]

        attn_J = (J_q @ J_k.transpose(-2, -1)) # [B, #heads, N, N]
        attn_R_linear = self.R_conv(relation_feature).reshape(B, N, N, H).permute(0, 3, 1, 2)  #[B, #heads, N, N]
        attn_R_qurt = (R_q.unsqueeze(-2) @ R_k.unsqueeze(-1)).squeeze() # [B, #heads, N, N]

        attn = (attn_J + attn_R_linear + attn_R_qurt) * self.scale #[B, #heads, N, N]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # [B, #heads, N, N]

        x = (attn @ J_v).transpose(1, 2).reshape(B, N, C) #[B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()


		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                
		self.norm_attn1 = norm_layer(dim)
		self.norm_attn2 = norm_layer(dim)
		self.norm_joint = norm_layer(dim)
                
		self.norm_relation1 = norm_layer(dim*4)
		self.norm_relation2 = norm_layer(dim)
        
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                
		self.mlp_joint = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
                
		self.mlp_relation1 = Mlp(in_features=dim*4, hidden_features=dim*4, out_features=dim, act_layer=act_layer, drop=drop)
		self.mlp_relation2 = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)



	def forward(self, joint_feature, relation_feature, mask=None):
		B, N, C = joint_feature.shape
        ## joint feature update through attention mechanism
		joint_feature = joint_feature + self.drop_path(self.attn(self.norm_attn1(joint_feature), self.norm_attn2(relation_feature), mask))
		joint_feature = joint_feature + self.drop_path(self.mlp_joint(self.norm_joint(joint_feature)))
		
        ## relation feature update 
        # M_ij = cat(J_i, J_j, R_ij, R_ji)
		joint_i = joint_feature.unsqueeze(1).repeat(1, N, 1, 1) #[B, N, N, D] J_i
		joint_j = joint_feature.unsqueeze(2).repeat(1, 1, N, 1) #[B, N, N, D] J_j
		relation_rev = relation_feature.swapaxes(-2, -3) #[B, N, N, D] E_ji
		relation_input = torch.cat((relation_feature, relation_rev, joint_i, joint_j), -1)
                
        # U_ij = R_ij + MLP(Norm(M_ij)), R_ij' = U_ij + MLP(Norm(U_ij))
		relation_feature = relation_feature + self.drop_path(self.mlp_relation1(self.norm_relation1(relation_input)))
		relation_feature = relation_feature + self.drop_path(self.mlp_relation2(self.norm_relation2(relation_feature)))
		return joint_feature, relation_feature

class Block2(nn.Module):

	def __init__(self, d_model, nhead, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
					 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()

		self.J = 13
		self.att = InterPersonAttention(d_model, nhead)

		self.norm_attn1 = norm_layer(d_model)
		self.norm_attn2 = norm_layer(d_model)
		self.norm_joint = norm_layer(d_model)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.mlp_joint = Mlp(in_features=d_model, hidden_features=d_model, act_layer=act_layer, drop=drop)



	def forward(self, person1, person2):

		person1_feature = person1 + self.drop_path(self.att(self.norm_attn1(person1), self.norm_attn2(person2)))
		person1_feature = person1_feature + self.drop_path(self.mlp_joint(self.norm_joint(person1)))

		person2_feature = person2 + self.drop_path(self.att(self.norm_attn2(person2), self.norm_attn1(person1)))
		person2_feature = person2_feature + self.drop_path(self.mlp_joint(self.norm_joint(person2)))
		# print(person1_feature.shape)
		# print(person2_feature.shape)
		# quit()
		# U_ij = R_ij + MLP(Norm(M_ij)), R_ij' = U_ij + MLP(Norm(U_ij))


		return person1_feature, person2_feature




class JRTransformer(nn.Module):
	def __init__(self, N=2, J=13, in_joint_size=16*6, in_relation_size=18, feat_size=128, out_joint_size=30*3, out_relation_size=30, num_heads=8, depth=4, norm_layer=nn.LayerNorm):
		super().__init__()

		self.J = 13
		self.input_frame = 16
		self.conv2d = nn.Conv2d(in_channels= 26, out_channels= 26,kernel_size=3,padding=1,stride=1)
		self.pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
		self.joint_encoder = MLP(in_joint_size, feat_size, (256, 256))
		self.relation_encoder = MLP(in_relation_size, feat_size, (256, 256))
		self.mlp = MLP(feat_size*2, feat_size, (256, 256))
		self.pe = PositionalEmbedding(N, J, feat_size)
		self.norm_layer = norm_layer(feat_size)
		self.Linear1 = nn.Linear(in_features=16,out_features=96)
		self.Linear2 = nn.Linear(in_features=16, out_features=96)
		self.GCNdecoer1 = nn.Linear(in_features=78,out_features=26)
		self.GCNdecoer2 = nn.Linear(in_features=78,out_features=26)
		self.Linear3 = nn.Linear(in_features=32,out_features=18)
		self.decoder = Decoder(input_dim=256, hidden_dim=1024, output_joints=13, output_dim=3)

		self.attn_encoder = nn.ModuleList([
			Block(feat_size, num_heads, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
			for i in range(depth)])


		self.attn_encoder_external = nn.ModuleList([
			Block2(feat_size, num_heads, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
			for i in range(depth)])
		
		#self.joint_decoder = MLP(feat_size, out_joint_size)
		self.GCN = GCN(input_feature = self.input_frame, hidden_feature = feat_size, p_dropout = 0.5, num_stage=1, node_n=78)
		self.relation_decoder = MLP(feat_size, out_relation_size)
		self.joint_decoder = MLP(feat_size, out_joint_size)
		self.initialize_weights()
 
	def initialize_weights(self):
		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		
		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)


	def forward(self, x_joint):
		B, NJ, T, D = x_joint.shape  #(128,26,16,6)
		   							 #(128,26,26,2)

		#x_joint = x_joint.view(B, NJ, -1)


		x_joint_person1 = x_joint[:, :13, :, :]
		x_joint_person2 = x_joint[:, 13:, :, :]
		x_joint_person11 = x_joint_person1.view(B, self.J, -1)       #(128,13,96)
		x_joint_person22 = x_joint_person2.view(B, self.J, -1)		#(128,13,96)
		x_joint_person1 = x_joint_person1.reshape(B, self.J*D, T)    #(128,78,16)
		x_joint_person2 = x_joint_person2.reshape(B, self.J*D, T)	 #(128,78,16)
		# x_joint_person1 = self.Linear1(x_joint_person1)
		# x_joint_person2 = self.Linear2(x_joint_person2)
		#
		x_person1_feature = self.joint_encoder(x_joint_person11)
		x_person2_feature = self.joint_encoder(x_joint_person22)

		#-------------------------------internal_interaction-------------------


		x_person1_internal = self.GCN(x_joint_person1)   #(128,78,128)
		x_person2_internal = self.GCN(x_joint_person2)   #(128,78,128)

		x_person1_internal = self.GCNdecoer1(x_person1_internal.permute(0, 2, 1)).permute(0,2,1)
		x_person2_internal = self.GCNdecoer2(x_person2_internal.permute(0, 2, 1)).permute(0,2,1)

		# print(x_person1_internal.shape)
		# print(x_person2_internal.shape)
		# quit()

		x_internal = torch.cat((x_person1_internal, x_person2_internal),dim=2)   #(128,26,52)
		x_internal = x_internal.unsqueeze(2).repeat_interleave(dim =2,repeats=26)
		# print(x_internal.shape)




		#-------------------------------------------------------------------

		#-----------------------------external_interaction-----------------------------
		pe_person1 = self.pe.forward_external()    #[13,128]
		pe_person2 = self.pe.forward_external()

		x_person1 = x_person1_feature
		x_person2 = x_person2_feature
		for i in range(len(self.attn_encoder_external)):
			blk2 = self.attn_encoder_external[i]
			x_person1 = x_person1 + pe_person1
			x_person2 = x_person2 + pe_person2
			x_person1,x_person2 = blk2(x_person1, x_person2)

		x_external = torch.cat((x_person1, x_person2),dim=1)

		x_external,x_internal = IAM(x_external,x_internal)


		pred = decoder(torch.cat(x_external,x_internal))


		#quit()
		return pred


