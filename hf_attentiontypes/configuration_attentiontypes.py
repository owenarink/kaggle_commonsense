from transformers import PretrainedConfig


class AttentionTypesConfig(PretrainedConfig):
    model_type = "attentiontypes"

    def __init__(
        self,
        vocab_size=12000,
        num_labels=1,
        pad_token_id=0,
        bos_token_id=3,
        eos_token_id=4,
        sep_token_id=2,
        unk_token_id=1,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        ff_mult=4,
        dropout=0.3,
        max_position_embeddings=512,
        pooling="mean",
        use_absolute_pos=False,
        tokenizer_class="PreTrainedTokenizerFast",
        grouped_max_len=128,
        bbpe_vocab_size=12000,
        bbpe_min_freq=2,
        ce_weight=1.0,
        hinge_weight=0.35,
        hinge_margin=0.2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.unk_token_id = unk_token_id
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.pooling = pooling
        self.use_absolute_pos = use_absolute_pos
        self.tokenizer_class = tokenizer_class
        self.grouped_max_len = grouped_max_len
        self.bbpe_vocab_size = bbpe_vocab_size
        self.bbpe_min_freq = bbpe_min_freq
        self.ce_weight = ce_weight
        self.hinge_weight = hinge_weight
        self.hinge_margin = hinge_margin
