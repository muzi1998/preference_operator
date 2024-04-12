import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=1) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def neg_feature(self, feature):
        feature = feature
        return feature

    def forward(self, logic):
        logic = 1. / logic
        return logic

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class KGAT(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,userpreference, A_in=None,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            # user_pre_embed = F.normalize(user_pre_embed, p=2, dim=1)
            # item_pre_embed = F.normalize(item_pre_embed, p=2, dim=1)
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False
        #logic
        self.userpreference = userpreference
        # self.fea2log = nn.Linear(248, 2 * 248, bias=False)
        self.fea2log = nn.Linear(248, 2 * 248)
        self.bn1 = nn.BatchNorm1d(248)
        self.bn2 = nn.BatchNorm1d(2 * 248)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmod = torch.nn.Sigmoid()
        self.projection_regularizer = Regularizer(0.05, 0.05, 1e9)
        self.Negation = Negation()
        self.center_net = BetaIntersection(248)
        self.logic_gamma = torch.tensor(0.)
        self.kl_weight = torch.tensor(0.001)

    def feature_to_beta(self, feature):

        logic_input = self.fea2log(self.bn1(feature))
        logic_input = self.sigmod(logic_input)
        # logic_input = self.bn2(logic_input)
        logic_input = self.projection_regularizer(logic_input)
        alpha, beta = torch.chunk(logic_input, 2, dim=-1)

        return alpha, beta

    def cal_logit_beta(self, entity_dist, path_dist):
        logit = self.logic_gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, path_dist), p=1, dim=-1)
        return logit

    def _construct_product_logic_embedding(self,embeds):
        item_alpha_input, item_beta_input = self.feature_to_beta(torch.tensor(embeds))
        item_logic_output = item_alpha_input / (item_alpha_input + item_beta_input)
        item_alpha_neg, item_beta_neg = self.Negation(item_alpha_input), self.Negation(item_beta_input)
        item_logic_neg_output = item_alpha_neg / (item_alpha_neg + item_beta_neg)
        return item_alpha_input, item_beta_input, item_logic_output, item_alpha_neg, item_beta_neg, item_logic_neg_output

    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed

    def calc_logic_KL_loss(self, user_ids, item_pos_ids, item_neg_ids,train_dist,item_alpha_input,item_beta_input,item_alpha_neg,item_beta_neg,all_embed):
        # KL_LOSS = torch.tensor(0.).to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
        KL_LOSS = []
        all_state_alpha = []
        all_state_beta = []
        for uid in user_ids:
            uid = int(uid.cpu().numpy())
            uid_state_alpha = []
            uid_state_beta = []
            for pid in train_dist[uid]:
                if pid in self.userpreference['user {}'.format(uid-self.n_entities)]['dislike']:
                    uid_state_alpha.append(item_alpha_neg[pid])
                    uid_state_beta.append(item_beta_neg[pid])
                else:
                    uid_state_alpha.append(item_alpha_input[pid])
                    uid_state_beta.append(item_beta_input[pid])
            # print(uid_state_alpha)
            if len(uid_state_alpha) > 0:
                for i in range(len(uid_state_alpha)):
                    if i == 0:
                        pa = uid_state_alpha[0]
                        pb = uid_state_beta[0]
                    elif i == 1:
                        pa = torch.vstack((uid_state_alpha[0], uid_state_alpha[1]))
                        pb = torch.vstack((uid_state_beta[0], uid_state_beta[1]))
                    elif i > 1:
                        pa = torch.vstack((pa, uid_state_alpha[i]))
                        pb = torch.vstack((pb, uid_state_beta[i]))
                if pa.dim() > 1:
                    state_pa, state_pb = self.center_net(pa, pb)
                else:
                    state_pa = pa
                    state_pb = pb
                state_dists = torch.distributions.beta.Beta(state_pa, state_pb)
                pos_logic_alpha,pos_logic_beta = item_alpha_input[item_pos_ids],item_beta_input[item_pos_ids]
                neg_logic_alpha,neg_logic_beta = item_alpha_input[item_neg_ids],item_beta_input[item_neg_ids]
                pos_act_logic_dicts = torch.distributions.beta.Beta(pos_logic_alpha, pos_logic_beta)
                neg_act_logic_dicts = torch.distributions.beta.Beta(neg_logic_alpha, neg_logic_beta)
                positive_logit = self.cal_logit_beta(pos_act_logic_dicts, state_dists)
                negative_logit = self.cal_logit_beta(neg_act_logic_dicts, state_dists)
                bpr = BPRLoss()
                act_kl_loss = bpr(positive_logit, negative_logit)
                KL_LOSS.append(act_kl_loss)
        return  torch.vstack(KL_LOSS).mean()


    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids,train_dist):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.calc_cf_embeddings()                       # (n_users + n_entities, concat_dim)
        item_alpha_input, item_beta_input, item_logic_output, item_alpha_neg, item_beta_neg, item_logic_neg_output = \
            self._construct_product_logic_embedding(all_embed) # (n_users + n_entities, concat_dim)
        user_embed = torch.mul(all_embed[user_ids],item_logic_output[user_ids])                            # (cf_batch_size, concat_dim)
        item_pos_embed = torch.mul(all_embed[item_pos_ids],item_logic_output[item_pos_ids])                  # (cf_batch_size, concat_dim)
        item_neg_embed = torch.mul(all_embed[item_neg_ids],item_logic_output[item_neg_ids])                      # (cf_batch_size, concat_dim)

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)
        #logic kl loss
        kl_loss = self.calc_logic_KL_loss(user_ids, item_pos_ids, item_neg_ids,train_dist,item_alpha_input,item_beta_input,item_alpha_neg,item_beta_neg,all_embed)

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss + self.cf_l2loss_lambda * kl_loss
        return loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score


    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)


