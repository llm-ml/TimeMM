# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

from .time_user_gate import build_user_time_features_from_train_df, TimeAwareUserModalWeight
from .time_edge_weight_builder_multiscale import MultiScaleTimeWeightCfg, build_edge_weight_for_ui_graph_multiscale
from .time_state_builder import TimeStateCfg, build_user_item_states_from_train_df
from .multiscale_order_smoothness_loss import MultiScaleOrderSmoothnessLoss, OrderSmoothnessCfg
from .loss_multiscale_complement import MultiScaleComplementLoss, MultiScaleCompCfg
from .loss_decision_space_align import DecisionSpaceAlignLoss, DecisionAlignCfg


class TIMEMM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TIMEMM, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        has_id = True
        self.tau_age_list = config["tau_age_list"]
        self.aux_weight = config["aux_weight"]
        self.use_age = config["use_age"]
        self.use_gap = config["use_gap"]
        self.transform = config["transform"]
        self.use_corr=config["use_corr"]
        self.use_var_floor=config["use_var_floor"]
        self.var_floor=config["var_floor"]
        self.var_weight=config["var_weight"]
        self.reg_weight1=config["reg_weight1"]
        self.reg_weight2=config["reg_weight2"]
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = config['num_layers']
        self.cold_start = 0
        self.dataset = dataset
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.mm_adj = None
        num_bands = len(self.tau_age_list)
        self.user_id_embedding = nn.Embedding(self.n_users, self.dim_latent)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim_latent)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.is_user_graph = False  # default flag

        user_emb_path = os.path.join(dataset_path, config['user_emb_file'])

        if os.path.isfile(user_emb_path):
            user_emb = np.load(user_emb_path, allow_pickle=True)
            self.user_emb = torch.from_numpy(user_emb).to(self.device)
            print(">>>>self.user_emb.shape=", self.user_emb.shape)

            indices, user_adj = self.get_knn_adj_mat(self.user_emb)
            self.user_adj = user_adj

            self.is_user_graph = True
        else:
            self.is_user_graph = False
            print(f">>>> user_emb_file not found, skip loading: {user_emb_path}")

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file) and False:
            self.mm_adj = torch.load(mm_adj_file)
            print(">>>>>Loaded from: ", mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
            print(">>>>>Save mm_adj to: ", mm_adj_file)
        self.use_item_graph = True
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # ===============================================================
        self.edge_weight = None
        # -------- train df --------
        df_all = dataset.dataset.df
        if hasattr(df_all, "columns") and ("x_label" in df_all.columns):
            df_train = df_all[df_all["x_label"] == 0].copy()
        else:
            df_train = df_all.copy()
        
        uid_field = dataset.dataset.uid_field
        iid_field = getattr(dataset.dataset, "iid_field", None)
        if iid_field is None or iid_field not in df_train.columns:
            iid_field = "itemID" if "itemID" in df_train.columns else iid_field
        assert iid_field is not None, "Cannot infer item id field (iid_field/itemID)."

        time_col = config["TIME_FIELD"]
        assert time_col in df_train.columns, f"TIME_FIELD={time_col} not found in df_train."

        self.num_scale = len(self.tau_age_list)
        tw_cfg = MultiScaleTimeWeightCfg(
            use_age=self.use_age,
            use_gap=self.use_gap,
            transform=self.transform,
            tau_age_list=self.tau_age_list,   # K
            clip_min=1e-3,
            clip_max=1.0,
        )
        self.edge_weight, self.edge_age_sec_1way, self.edge_gap_sec_1way = build_edge_weight_for_ui_graph_multiscale(
            edge_index_1way_np=edge_index,
            num_user=self.num_user,
            device=self.device,
            df_train=df_train,
            uid_field=uid_field,
            iid_field=iid_field,
            time_col=time_col,
            cfg=tw_cfg,
        )

        # time feature buffer
        user_time_feat = build_user_time_features_from_train_df(df_train, uid_field, time_col, self.num_user)
        self.register_buffer("user_time_feat", user_time_feat.to(self.device))

        # time->weight module
        self.time_user_gate = TimeAwareUserModalWeight(in_dim=self.user_time_feat.size(1), hidden=32, alpha=0.2)
        self.use_time_user_gate = bool(True)
        # df_train
        ts_cfg = TimeStateCfg(transform="log1p", include_std=True)

        self.user_state, self.item_state = build_user_item_states_from_train_df(
            df_train=df_train,
            uid_field=uid_field,
            iid_field=iid_field,
            time_col=time_col,
            num_user=self.num_user,
            num_item=self.num_item,
            device=self.device,
            cfg=ts_cfg,
        )
        # pdb.set_trace()
        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx), self.v_feat.size(1)).to(self.device)
            self.visual_spectralfilter = TemporalSpectralFilter(self.dataset, batch_size, num_bands, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx), self.t_feat.size(1)).to(self.device)
            self.textual_spectralfilter = TemporalSpectralFilter(self.dataset, batch_size, num_bands, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)
        
        self.id_spectralfilter = TemporalSpectralFilter(self.dataset, batch_size, num_bands, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,
                             device=self.device, features=self.item_id_embedding.weight)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        self.order_smooth_loss = MultiScaleOrderSmoothnessLoss(
            OrderSmoothnessCfg(
                short_to_long=True,    
                margin=0.1,             
                normalize_by_dim=True,
                reduction="sum",
            )
        )
        
        # fintune hyper-param
        self.comp_loss = MultiScaleComplementLoss(
            MultiScaleCompCfg(
                use_corr=self.use_corr,
                use_var_floor=self.use_var_floor,
                var_floor=self.var_floor,
                var_weight=self.var_weight,
            )
        )
        # Optional
        self.dec_align_cfg = DecisionAlignCfg(
            mode="teacher_student", 
            temperature=0.5,
            teacher_index=-1,     
            detach_teacher=True,
        )
        self.dec_align_loss = DecisionSpaceAlignLoss(self.dec_align_cfg)


    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        if self.is_user_graph:
            self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
            self.user_weight_matrix = self.user_weight_matrix.to(self.device)
        pass

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))
    
    def forward(self):
        base_representation = None

        # ID modality
        self.id_rep, id_rep_per_spectral, self.id_preference = self.id_spectralfilter(
            self.edge_index_dropv,
            self.edge_index,
            self.item_id_embedding.weight,
            edge_weight=self.edge_weight,
            user_state=self.user_state,
            item_state=self.item_state,
        )

        # visual modality
        if self.v_feat is not None:
            self.v_rep, v_rep_per_spectral, self.v_preference = self.visual_spectralfilter(
                self.edge_index_dropv,
                self.edge_index,
                self.image_embedding.weight,
                edge_weight=self.edge_weight,
                user_state=self.user_state,
                item_state=self.item_state,
            )
            base_representation = self.v_rep
        else:
            v_rep_per_spectral = None

        # textual modality
        if self.t_feat is not None:
            self.t_rep, t_rep_per_spectral, self.t_preference = self.textual_spectralfilter(
                self.edge_index_dropt,
                self.edge_index,
                self.text_embedding.weight,
                edge_weight=self.edge_weight,
                user_state=self.user_state,
                item_state=self.item_state,
            )

            if base_representation is None:
                base_representation = self.t_rep
            else:
                # Keep original behavior: once both V and T exist, use concatenation of (ID, V, T)
                base_representation = torch.cat((self.id_rep, self.v_rep, self.t_rep), dim=1)
        else:
            t_rep_per_spectral = None

        # Build routed representation
        user_embedding, gate_weights = self.build_modality_routing()

        # item embedding comes from the spectral filter
        item_embedding = base_representation[self.num_user:]

        # user item graph propagation
        user_emb_multi, item_emb_multi = self.propagate_user_item_graphs(
            user_embedding, item_embedding
        )

        per_spectral_rep_list = [id_rep_per_spectral, v_rep_per_spectral, t_rep_per_spectral]
        per_spectral_rep_list = torch.cat(per_spectral_rep_list, dim=-1)

        return user_emb_multi, item_emb_multi, per_spectral_rep_list, gate_weights, 0.0

    def cross_entropy_loss(self, u_emb, pos_i_emb, neg_i_emb):
        """
        Treat (user, pos_item) as positive samples and (user, neg_item) as negative samples,
        then concatenate them and optimize using a binary cross-entropy loss.
        """
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)  # [batch_size]
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)  # [batch_size]

        pos_labels = torch.ones_like(pos_scores)  # [batch_size],
        neg_labels = torch.zeros_like(neg_scores) # [batch_size],

        all_scores = torch.cat([pos_scores, neg_scores], dim=0)   # [2 * batch_size]
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)   # [2 * batch_size]

        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
        
        return loss

    def calculate_loss(self, interaction):
        # parse interaction
        batch_user_idx = interaction[0]
        batch_pos_item_idx = interaction[1]
        batch_neg_item_idx = interaction[2]
        #
        all_user_emb, all_item_emb, scale_rep_tensor, gate_weights, _ = self.forward()

        # split scale-wise representations
        scale_user_rep = scale_rep_tensor[:, : self.num_user, :]
        scale_item_rep = scale_rep_tensor[:, self.num_user :, :]

        # each element is for a given spectral
        scale_user_rep_list = list(scale_user_rep.unbind(dim=0))
        scale_item_rep_list = list(scale_item_rep.unbind(dim=0))

        # Gather per-spectral embeddings for the batch
        batch_user_rep_scales = [rep[batch_user_idx] for rep in scale_user_rep_list]
        batch_pos_item_rep_scales = [rep[batch_pos_item_idx] for rep in scale_item_rep_list]
        batch_neg_item_rep_scales = [rep[batch_neg_item_idx] for rep in scale_item_rep_list]

        # auxiliary losses
        smoothness_loss = self.order_smooth_loss(batch_user_rep_scales, batch_pos_item_rep_scales)  # already satisfied
        aux_loss = self.comp_loss(batch_user_rep_scales, batch_pos_item_rep_scales, batch_neg_item_rep_scales)

        # main loss
        batch_user_emb = all_user_emb[batch_user_idx]
        batch_pos_item_emb = all_item_emb[batch_pos_item_idx]
        batch_neg_item_emb = all_item_emb[batch_neg_item_idx]

        mf_loss = self.cross_entropy_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)

        # regularization
        reg_term = self.regularization(gate_weights)

        return mf_loss #+ aux_loss * self.aux_weight + reg_term 
    

    def propagate_user_item_graphs(self, user_embedding, item_embedding):
        """
        Perform optional user-side and item-side graph propagation, then fuse with the
        original embeddings by residual addition.
        """
        # Item-side graph propagation
        if self.use_item_graph:
            propagated_item_emb = item_embedding
            for _ in range(self.n_layers):
                propagated_item_emb = torch.sparse.mm(self.mm_adj, propagated_item_emb)
        else:
            propagated_item_emb = item_embedding

        #User-side input fusion
        fused_user_input = torch.cat(
            (self.id_preference, self.v_preference, self.t_preference), dim=1
        )  # (num_users, dim_total)
        self.all_preference = fused_user_input

        #User-side graph propagation
        if self.is_user_graph:
            propagated_user_emb = fused_user_input
            for _ in range(self.n_layers):
                propagated_user_emb = torch.sparse.mm(self.user_adj, propagated_user_emb)
        else:
            propagated_user_emb = fused_user_input
        user_emb_fused = user_embedding + propagated_user_emb
        item_emb_fused = item_embedding + propagated_item_emb
        return user_emb_fused, item_emb_fused


    def build_modality_routing(self):
        """
        return routed representations Tensor [num_user, 3 * rep_dim]
        router_weights: Tensor returned by time_user_gate (typically [num_user, 3, 1])
        """
        # Ensure reps are 3D: [N, D] -> [N, D, 1]
        id_rep_expanded = torch.unsqueeze(self.id_rep, 2)
        visual_rep_expanded = torch.unsqueeze(self.v_rep, 2)
        text_rep_expanded = torch.unsqueeze(self.t_rep, 2)

        # Stack
        rep_stacked = torch.cat(
            (
                id_rep_expanded[: self.num_user],
                visual_rep_expanded[: self.num_user],
                text_rep_expanded[: self.num_user],
            ),
            dim=2,
        )

        # compute router weights
        router_weights = self.time_user_gate(self.user_time_feat)

        # apply gate: [U, 1, 3] * [U, D, 3] -> [U, D, 3]
        routed_rep = router_weights.transpose(1, 2) * rep_stacked

        # flatten: [U, D, 3] -> [U, 3D]
        routed_rep = torch.cat(
            (routed_rep[:, :, 0], routed_rep[:, :, 1], routed_rep[:, :, 2]),
            dim=1,
        )

        return routed_rep, router_weights
    
    def regularization(self, weight_u, reg_gate_weight=None, reg_loss_weight=None, eps=1e-12):
        """
        Regularization terms for gating weights and embedding and preferences.
        """
        if reg_gate_weight is None:
            reg_gate_weight = getattr(self, "reg_weight1", 1e-3)
        if reg_loss_weight is None:
            reg_loss_weight = getattr(self, "reg_weight2", 1e-1)

        # Gate entropy regularization
        gate_prob = weight_u.squeeze(-1)                               # [U, 3] if input is [U, 3, 1]
        gate_prob = gate_prob.clamp_min(eps)                           # avoid log(0)
        entropy = -(gate_prob * gate_prob.log()).sum(dim=1).mean()     # mean entropy
        reg_gate = reg_gate_weight * entropy                           # negative => encourage higher entropy

        # L2 regularization
        reg_l2 = (
            self.all_preference.pow(2).mean()
            + self.user_id_embedding.weight.pow(2).mean()
            + self.item_id_embedding.weight.pow(2).mean()
        )
        total_reg = reg_gate + reg_loss_weight * reg_l2
        return total_reg


    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, _, _, _ = self.forward()
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class TemporalSpectralFilter(torch.nn.Module):
    def __init__(
        self,datasets,batch_size,num_bands,num_user,num_item,dim_id,aggr_mode,num_layer,has_id,dropout,dim_latent=None,device=None,features=None,u_state_dim=7,i_state_dim=7):
        super(TemporalSpectralFilter, self).__init__()

        # Core configs
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        # User preference and feature projector
        if self.dim_latent:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.tensor(
                        np.random.randn(num_user, self.dim_latent),
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                    gain=1,
                ).to(self.device)
            )
            self.feature_mlp = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.feature_mlp_out = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_layer = GraphConvolutionNetwork(
                self.dim_latent, self.dim_latent, aggr=self.aggr_mode
            )
        else:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.tensor(
                        np.random.randn(num_user, self.dim_feat),
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                    gain=1,
                ).to(self.device)
            )
            # Keep the original behavior, including the original argument usage
            self.conv_embed_layer = GraphConvolutionNetwork(
                self.dim_latent, self.dim_latent, aggr=self.aggr_mode
            )

        self.K = num_bands  # len(tau_age_list)
        user_state_dim = u_state_dim
        item_state_dim = i_state_dim

        gate_mode = "shared"
        if gate_mode == "shared":
            # Option A: shared gate
            self.time_scale_gate = TimeScaleGate(
                K=self.K,
                user_in=user_state_dim,
                item_in=item_state_dim,
                hidden=64,
                mode="shared",
                temperature=0.7,
                dropout=0.0,
            ).to(self.device)
        else:
            # Option B: separate gates
            self.time_scale_gate = TimeScaleGate(
                K=self.K,
                user_in=user_state_dim,
                item_in=item_state_dim,
                hidden=64,
                mode="separate",
                temperature=0.7,
                dropout=0.0,
            ).to(self.device)

    def forward(
        self,
        edge_index_drop,
        edge_index,
        features,
        edge_weight,
        user_state,
        item_state,
    ):
        """
        edge_weight multi-scale:  [K, 2*nnz]
        Output filter_emb fused representation filter_emb_hat_k: [K, N, D] per-scale representation, and preference
        """
        projected_features = (
            self.feature_mlp_out(F.leaky_relu(self.feature_mlp(features)))
            if self.dim_latent
            else features
        )

        node_init = torch.cat((self.preference, projected_features), dim=0).to(self.device)
        node_init = F.normalize(node_init).to(self.device)

        # ensure edge_weight is 2D: [K, 2*nnz]
        if edge_weight is None:
            edge_weight_per_scale = None
            num_scales = 1
        else:
            if edge_weight.dim() == 1:
                edge_weight_per_scale = edge_weight.unsqueeze(0)  # [1, 2*nnz]
                num_scales = 1
            elif edge_weight.dim() == 2:
                edge_weight_per_scale = edge_weight              # [K, 2*nnz]
                num_scales = edge_weight.size(0)
            else:
                raise ValueError(
                    f"edge_weight must be 1D or 2D, got shape={tuple(edge_weight.shape)}"
                )

        # run spectral propagations
        per_scale_embeddings = []
        for scale_idx in range(num_scales):
            scale_edge_weight = (
                None
                if edge_weight_per_scale is None
                else edge_weight_per_scale[scale_idx]
            )

            layer_outputs = [node_init]
            hidden = node_init
            for _ in range(self.num_layer):
                hidden = self.conv_embed_layer(hidden, edge_index, scale_edge_weight)
                layer_outputs.append(hidden)

            scale_embedding = sum(layer_outputs)  # [N, D]
            per_scale_embeddings.append(scale_embedding)

        filter_emb_hat_k = torch.stack(per_scale_embeddings, dim=0)  # [K, N, D]

        # filter [N, K]
        gate_scores = self.time_scale_gate(user_state, item_state)
        gate_kn1 = gate_scores.t().unsqueeze(-1)         # [K, N, 1]
        filter_emb = (filter_emb_hat_k * gate_kn1).sum(dim=0)          # [N, D]

        return filter_emb, filter_emb_hat_k, self.preference

class TimeScaleGate(nn.Module):
    """
    Produce per-node mixture weights over K time-scales.
    user_state [U, Fu] item_state: [I, Fi]
    Output gate [N, K] where N=U+I and sum_k gate[n,k]=1
    """
    def __init__(
        self,
        K: int,
        user_in: int,
        item_in: int,
        hidden: int = 64,
        mode: str = "shared",   # "shared" or "separate", adaptively change
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert mode in {"shared", "separate"}
        self.K = int(K)
        self.mode = mode
        self.temperature = float(max(temperature, 1e-6))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mode == "shared":
            self.user_stem = nn.Sequential(nn.Linear(user_in, hidden), nn.ReLU())
            self.item_stem = nn.Sequential(nn.Linear(item_in, hidden), nn.ReLU())
            self.head = nn.Linear(hidden, K)

        else:
            self.user_gate = nn.Sequential(
                nn.Linear(user_in, hidden), nn.ReLU(),
                nn.Linear(hidden, K)
            )
            self.item_gate = nn.Sequential(
                nn.Linear(item_in, hidden), nn.ReLU(),
                nn.Linear(hidden, K)
            )

    def forward(self, user_state: torch.Tensor, item_state: torch.Tensor) -> torch.Tensor:
        if self.mode == "shared":
            u = self.drop(self.user_stem(user_state))         # [U, H]
            i = self.drop(self.item_stem(item_state))         # [I, H]
            u_logits = self.head(u)                           # [U, K]
            i_logits = self.head(i)                           # [I, K]
        else:
            u_logits = self.drop(self.user_gate(user_state))  # [U, K]
            i_logits = self.drop(self.item_gate(item_state))  # [I, K]

        logits = torch.cat([u_logits, i_logits], dim=0)       # [N, K]
        gate = torch.softmax(logits / self.temperature, dim=1)
        return gate


class GraphConvolutionNetwork(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        bias=True,
        aggr="add",
        **kwargs
    ):
        super(GraphConvolutionNetwork, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    # Make edge_weight optional and pass it through to propagate()
    def forward(self, node_features, edge_index, edge_weight=None, size=None):
        if size is None:
            # remove_self_loops must take edge_weight together,
            # otherwise the number of edges will mismatch.
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Ensure node_features is at least 2D: [num_nodes] -> [num_nodes, 1]
        node_features = (
            node_features.unsqueeze(-1) if node_features.dim() == 1 else node_features
        )

        return self.propagate(
            edge_index,
            size=(node_features.size(0), node_features.size(0)),
            x=node_features,
            edge_weight=edge_weight,  # pass through
        )

    # message() receives edge_weight as an argument
    def message(self, x_j, edge_index, size, edge_weight):
        # Keep the original behavior, only apply normalization when aggr == "add"
        if self.aggr != "add":
            return x_j

        src_index, dst_index = edge_index  # src -> dst

        # No edge weights: keep the original unweighted GCN normalization
        if edge_weight is None:
            out_degree = degree(src_index, size[0], dtype=x_j.dtype)
            out_degree_inv_sqrt = out_degree.clamp(min=1e-12).pow(-0.5)
            norm_coef = out_degree_inv_sqrt[src_index] * out_degree_inv_sqrt[dst_index]
            return norm_coef.view(-1, 1) * x_j

        # With edge weights, weighted GCN normalization
        edge_weight_typed = edge_weight.to(dtype=x_j.dtype)

        weighted_out_degree = torch.zeros(
            size[0], device=src_index.device, dtype=x_j.dtype
        )
        # weighted_out_degree[u] = sum_out w(u, *)
        weighted_out_degree.scatter_add_(0, src_index, edge_weight_typed)

        weighted_out_degree_inv_sqrt = weighted_out_degree.clamp(min=1e-12).pow(-0.5)

        # norm_w = w_ij / sqrt(deg_w[i] * deg_w[j])
        weighted_norm_coef = (
            weighted_out_degree_inv_sqrt[src_index]
            * edge_weight_typed
            * weighted_out_degree_inv_sqrt[dst_index]
        )

        # Debug
        # raise
        return weighted_norm_coef.view(-1, 1) * x_j

    def update(self, aggregated_messages):
        return aggregated_messages

    def __repr__(self):
        return "{}({},{})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
