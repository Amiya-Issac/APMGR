import torch
import torch.nn.functional as F
import torch.nn as nn


class LookupEmbedding(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


class MetaNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, meta_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class PersonalizedBridge(nn.Module):
    def __init__(self, emb_dim, meta_dim, pad_idx,tgt_gating):
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.tgt_gating = tgt_gating
        self.event_K = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1, False)
        )
        self.event_softmax = nn.Softmax(dim=1)
        self.meta_net_matrix = MetaNetwork(emb_dim, emb_dim * emb_dim, meta_dim)
        self.target_adapter = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(emb_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, emb_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.feature_gate[2].bias, 3.0)
        nn.init.xavier_uniform_(self.feature_gate[2].weight)

    def forward(self, uid_emb_src, emb_fea_src, seq_index_src, emb_tgt_item):
        batch_size = uid_emb_src.size(0)
        mask = (seq_index_src == self.pad_idx).float()
        event_K = self.event_K(emb_fea_src)
        t = event_K - mask.unsqueeze(2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea_src, 1)
        bridge_params = self.meta_net_matrix(his_fea)
        bridge_matrix = bridge_params.view(batch_size, self.emb_dim, self.emb_dim)
        base_user_emb = torch.bmm(uid_emb_src, bridge_matrix).squeeze(1)
        if self.tgt_gating:
            tgt_vec = self.target_adapter(emb_tgt_item.squeeze(1))
            gate = self.feature_gate(tgt_vec)
            final_user_emb = base_user_emb * gate
        else:
            final_user_emb = base_user_emb
        return final_user_emb


class PrototypeLearner(nn.Module):

    def __init__(self, emb_dim, meta_dim, num_prototypes=20, use_dynamicDelta=True, pad_idx=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_prototypes = num_prototypes
        self.use_dynamicDelta = use_dynamicDelta
        self.pad_idx = pad_idx

        self.base_proto_src = nn.Parameter(torch.randn(num_prototypes, emb_dim) * 0.01)
        self.base_proto_tgt = nn.Parameter(torch.randn(num_prototypes, emb_dim) * 0.01)

        self.W_user = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_item = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_proto_src = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_proto_tgt = nn.Linear(emb_dim, emb_dim, bias=False)

        nn.init.xavier_uniform_(self.base_proto_src)
        nn.init.xavier_uniform_(self.base_proto_tgt)

        self.event_K = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1, False)
        )
        self.event_softmax = nn.Softmax(dim=1)
        self.W_u = nn.Linear(emb_dim, num_prototypes, bias=False)
        self.W_t = nn.Linear(emb_dim, num_prototypes, bias=False)
        if self.use_dynamicDelta:
            self.meta_network = MetaNetwork(
                input_dim=emb_dim * 2,
                output_dim=num_prototypes * emb_dim * 2,
                meta_dim=meta_dim
            )

    def shifted_cosine_similarity(self, A, B):
        if A.dim() == 2:
            A = A.unsqueeze(1)
        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        cosine_sim = torch.bmm(A_norm, B_norm.transpose(1, 2))
        return (1.0+cosine_sim).squeeze(1)

    def forward(self, uid_emb_src, iid_emb_tgt, emb_fea_src=None, seq_index_src=None):
        batch_size = uid_emb_src.size(0)

        mask = (seq_index_src == self.pad_idx).float()
        event_K = self.event_K(emb_fea_src)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea_src, 1)
        if self.use_dynamicDelta:
            meta_input = torch.cat([his_fea, iid_emb_tgt], dim=-1)
            proto_params = self.meta_network(meta_input)
            proto_params = proto_params.view(batch_size, self.num_prototypes, self.emb_dim * 2)
            delta_src = proto_params[:, :, :self.emb_dim]
            delta_tgt = proto_params[:, :, self.emb_dim:]
            proto_src = self.base_proto_src.unsqueeze(0) + delta_src
            proto_tgt = self.base_proto_tgt.unsqueeze(0) + delta_tgt
        else:
            proto_src = self.base_proto_src.unsqueeze(0).expand(batch_size, -1, -1)
            proto_tgt = self.base_proto_tgt.unsqueeze(0).expand(batch_size, -1, -1)

        user_sim = self.shifted_cosine_similarity(his_fea, proto_src)
        item_sim = self.shifted_cosine_similarity(iid_emb_tgt, proto_tgt)
        # score = torch.sum(user_sim * item_sim, dim=1)
        u_hat = self.W_u(his_fea.squeeze(1))  # [batch, L_t]
        t_hat = self.W_t(iid_emb_tgt.squeeze(1))  # [batch, L_u]

        score = torch.sum(user_sim * t_hat, dim=1) + torch.sum(item_sim * u_hat, dim=1)
        return score


class APMGR(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim=10, meta_dim=50,
                 num_prototypes=20, base_model='MF',
                 use_personalized=True, use_prototype=True, tgt_gating=True,
                 use_dynamicDelta=True):
        super().__init__()
        self.uid_all = uid_all
        self.iid_all = iid_all
        self.emb_dim = emb_dim
        self.meta_dim = meta_dim
        self.base_model = base_model
        self.use_personalized = use_personalized
        self.use_prototype = use_prototype
        self.use_dynamicDelta = use_dynamicDelta
        self.tgt_gating = tgt_gating
        self.num_prototypes = num_prototypes

        if base_model == 'MF':
            self.src_model = MFBase(uid_all, iid_all, emb_dim)
            self.tgt_model = MFBase(uid_all, iid_all, emb_dim)
        elif base_model == 'GMF':
            self.src_model = GMFBase(uid_all, iid_all, emb_dim)
            self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        elif base_model == 'DNN':
            self.src_model = DNNBase(uid_all, iid_all, emb_dim)
            self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        else:
            raise ValueError(f'Unknown base model: {base_model}')

        if self.use_personalized:
            self.personalized_bridge = PersonalizedBridge(emb_dim, meta_dim, pad_idx=iid_all,tgt_gating = self.tgt_gating)
        else:
            self.personalized_bridge = None

        if self.use_prototype:
            self.prototype_learner = PrototypeLearner(
                emb_dim, meta_dim,
                num_prototypes=self.num_prototypes,
                use_dynamicDelta=self.use_dynamicDelta,
                pad_idx=iid_all,
            )
        else:
            self.prototype_learner = None

        if self.use_personalized and self.use_prototype:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, stage):
        if stage == 'train_src':
            return self.src_model.forward(x)
        elif stage in ['train_tgt', 'test_tgt']:
            return self.tgt_model.forward(x)
        elif stage in ['train_cross', 'test_cross']:
            iid_emb_tgt = self.tgt_model.get_item_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src_raw = self.src_model.get_user_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.get_item_embedding(x[:, 2:])
            seq_index = x[:, 2:]

            score_personal = None
            score_proto = None

            if self.use_personalized:
                user_emb = self.personalized_bridge(uid_emb_src_raw, ufea, x[:, 2:], iid_emb_tgt)
                if self.base_model == 'MF':
                    score_personal = torch.sum(user_emb * iid_emb_tgt.squeeze(1), dim=1)
                elif self.base_model == 'GMF':
                    product = user_emb * iid_emb_tgt.squeeze(1)
                    score_personal = self.tgt_model.linear(product).squeeze(1)
                else:
                    score_personal = torch.sum(self.tgt_model.linear(user_emb) * iid_emb_tgt.squeeze(1), dim=1)

            if self.use_prototype:
                score_proto = self.prototype_learner(
                    uid_emb_src_raw.squeeze(1), iid_emb_tgt.squeeze(1),
                    emb_fea_src=ufea, seq_index_src=seq_index)

            if self.use_personalized and self.use_prototype:
                w = torch.sigmoid(self.fusion_weight)
                output = w * score_proto + (1 - w) * score_personal
            elif self.use_personalized:
                output = score_personal
            else:
                output = score_proto

            return output


class MFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        return torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)

    def get_user_embedding(self, uid):
        return self.embedding.uid_embedding(uid)

    def get_item_embedding(self, iid):
        return self.embedding.iid_embedding(iid)


class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        product = emb[:, 0, :] * emb[:, 1, :]
        return self.linear(product).squeeze(1)

    def get_user_embedding(self, uid):
        return self.embedding.uid_embedding(uid)

    def get_item_embedding(self, iid):
        return self.embedding.iid_embedding(iid)


class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        return torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)

    def get_user_embedding(self, uid):
        return self.linear(self.embedding.uid_embedding(uid))

    def get_item_embedding(self, iid):
        return self.embedding.iid_embedding(iid)