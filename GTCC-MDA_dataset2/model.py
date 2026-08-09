from torch_geometric.nn import GCN2Conv

try:
    from graphormer_dataset1.layers import *
except ModuleNotFoundError:
    from layers import *


# Fixed architecture switches for reproducible ablation variants.
MODEL_USES_LOCAL = True
MODEL_USES_GLOBAL = True
MODEL_USES_CONTRAST = True


class CollaborativeContrastiveLoss(nn.Module):
    """Multi-positive, bidirectional InfoNCE for local/global node views."""

    def __init__(self, local_dim, global_dim, projection_dim, temperature=0.2,
                 balance=0.5, dropout=0.1):
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= balance <= 1:
            raise ValueError("balance must be in [0, 1]")
        self.temperature = temperature
        self.balance = balance
        self.local_projector = self._projector(
            local_dim, projection_dim, dropout)
        self.global_projector = self._projector(
            global_dim, projection_dim, dropout)

    @staticmethod
    def _projector(input_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def _directional_loss(logits, positive_mask, candidate_mask):
        min_value = torch.finfo(logits.dtype).min
        positives = logits.masked_fill(~positive_mask, min_value)
        candidates = logits.masked_fill(~candidate_mask, min_value)
        return -(
            torch.logsumexp(positives, dim=1)
            - torch.logsumexp(candidates, dim=1)
        ).mean()

    def forward(self, local_features, global_features, positive_mask,
                candidate_mask=None):
        local = F.normalize(self.local_projector(local_features), dim=-1)
        global_ = F.normalize(self.global_projector(global_features), dim=-1)
        logits = local @ global_.t() / self.temperature
        positive_mask = positive_mask.to(
            device=logits.device, dtype=torch.bool)
        candidate_mask = (
            torch.ones_like(positive_mask)
            if candidate_mask is None
            else candidate_mask.to(device=logits.device, dtype=torch.bool)
        )
        expected_shape = (logits.size(0), logits.size(1))
        if (positive_mask.shape != expected_shape
                or candidate_mask.shape != expected_shape):
            raise ValueError(
                f"contrastive masks must have shape {expected_shape}, got "
                f"{tuple(positive_mask.shape)} and "
                f"{tuple(candidate_mask.shape)}"
            )
        positive_mask = positive_mask & candidate_mask
        if not positive_mask.any(dim=1).all():
            raise ValueError(
                "every node must have at least one contrastive positive")
        local_to_global = self._directional_loss(
            logits, positive_mask, candidate_mask)
        global_to_local = self._directional_loss(
            logits.t(), positive_mask.t(), candidate_mask.t())
        return (
            self.balance * local_to_global
            + (1.0 - self.balance) * global_to_local
        )


class TransformerModel(nn.Module):
    """Local/global link predictor with interpretable late fusion."""

    def __init__(
            self,
            hops,
            output_dim,
            input_dim,
            pe_dim,
            num_drug,
            num_meta,
            graphformer_layers,
            num_heads,
            node_feature_dim,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            attention_dropout_rate,
            GCNII_layers,
            gcn_hidden_dim,
            gcnii_alpha,
            gcnii_theta,
            contrast_dim,
            contrast_temperature,
            contrast_balance,
            decoder_hidden_dim,
    ):
        super().__init__()
        self.use_local = MODEL_USES_LOCAL
        self.use_global = MODEL_USES_GLOBAL
        self.use_contrast = MODEL_USES_CONTRAST
        if not self.use_local and not self.use_global:
            raise ValueError("at least one model branch must be enabled")
        if self.use_contrast and not (self.use_local and self.use_global):
            raise ValueError(
                "contrastive learning requires both local and global branches")

        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_meta = num_meta
        self.num_drug = num_drug
        self.dropout_rate = dropout_rate

        if self.use_local:
            self.local_input_projection = (
                nn.Identity()
                if node_feature_dim == gcn_hidden_dim
                else nn.Linear(node_feature_dim, gcn_hidden_dim)
            )
            self.convs = nn.ModuleList([
                GCN2Conv(
                    channels=gcn_hidden_dim,
                    alpha=gcnii_alpha,
                    theta=gcnii_theta,
                    layer=layer + 1,
                )
                for layer in range(GCNII_layers)
            ])
            self.local_output_projection = nn.Sequential(
                nn.Linear(gcn_hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.local_decoder = PairMLPDecoder(
                embedding_dim=output_dim,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout_rate,
            )

        if self.use_global:
            self.att_embeddings_nope = nn.Linear(input_dim, hidden_dim)
            self.layers = nn.ModuleList([
                EncoderLayer(
                    hidden_dim,
                    ffn_dim,
                    dropout_rate,
                    attention_dropout_rate,
                    num_heads,
                )
                for _ in range(graphformer_layers)
            ])
            self.final_ln = nn.LayerNorm(hidden_dim)
            self.attn_layer = nn.Linear(2 * hidden_dim, 1)
            self.global_output_projection = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.global_decoder = PairMLPDecoder(
                embedding_dim=output_dim,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout_rate,
            )

        if self.use_local and self.use_global:
            # Softmax([0, 0]) starts with equal local/global weights.
            self.fusion_logits = nn.Parameter(torch.zeros(2))

        if self.use_contrast:
            self.contrastive = CollaborativeContrastiveLoss(
                gcn_hidden_dim,
                hidden_dim,
                contrast_dim,
                contrast_temperature,
                contrast_balance,
                dropout_rate,
            )

        self.apply(
            lambda module: init_params(
                module, n_layers=max(graphformer_layers, 1)))

    def _encode_local(self, meta_data, drug_data):
        x_0_meta = self.local_input_projection(meta_data.x)
        x_meta = x_0_meta
        for conv in self.convs:
            x_meta = conv(
                x_meta,
                x_0_meta,
                meta_data.edge_index,
                edge_weight=meta_data.edge_weight,
            )

        x_0_drug = self.local_input_projection(drug_data.x)
        x_drug = x_0_drug
        for conv in self.convs:
            x_drug = conv(
                x_drug,
                x_0_drug,
                drug_data.edge_index,
                edge_weight=drug_data.edge_weight,
            )
        return torch.cat((x_meta, x_drug), dim=0)

    def _encode_global(self, processed_features):
        tensor = self.att_embeddings_nope(processed_features)
        for encoder in self.layers:
            tensor = encoder(tensor)
        x_former = self.final_ln(tensor)

        node_tensor = x_former[:, :1, :]
        if self.seq_len == 1:
            return node_tensor.squeeze(1)

        neighbor_tensor = x_former[:, 1:, :]
        target = node_tensor.expand(-1, self.seq_len - 1, -1)
        attention = self.attn_layer(
            torch.cat((target, neighbor_tensor), dim=2))
        attention = F.softmax(attention, dim=1)
        neighbor_summary = torch.sum(
            neighbor_tensor * attention, dim=1, keepdim=True)
        return (node_tensor + neighbor_summary).squeeze(1)

    def forward(
            self,
            processed_features,
            meta_data,
            drug_data,
            pair_edge_index,
            positive_mask=None,
            candidate_mask=None,
            return_details=False,
    ):
        local_features = None
        global_features = None
        local_logits = None
        global_logits = None

        if self.use_local:
            if meta_data is None or drug_data is None:
                raise ValueError(
                    "local branch requires meta_data and drug_data")
            local_features = self._encode_local(meta_data, drug_data)
            local_embeddings = self.local_output_projection(local_features)
            local_logits = self.local_decoder(
                local_embeddings, pair_edge_index, self.num_meta)

        if self.use_global:
            if processed_features is None:
                raise ValueError(
                    "global branch requires processed_features")
            global_features = self._encode_global(processed_features)
            global_embeddings = self.global_output_projection(global_features)
            global_logits = self.global_decoder(
                global_embeddings, pair_edge_index, self.num_meta)

        if self.use_local and self.use_global:
            fusion_weights = torch.softmax(self.fusion_logits, dim=0)
            fused_logits = (
                fusion_weights[0] * local_logits
                + fusion_weights[1] * global_logits
            )
        elif self.use_local:
            fusion_weights = local_logits.new_tensor([1.0, 0.0])
            fused_logits = local_logits
        else:
            fusion_weights = global_logits.new_tensor([0.0, 1.0])
            fused_logits = global_logits

        contrastive_loss = None
        if self.use_contrast and positive_mask is not None:
            contrastive_loss = self.contrastive(
                local_features,
                global_features,
                positive_mask,
                candidate_mask,
            )

        if not return_details:
            return fused_logits
        return {
            "fused_logits": fused_logits,
            "local_logits": local_logits,
            "global_logits": global_logits,
            "fusion_weights": fusion_weights,
            "contrastive_loss": contrastive_loss,
        }
