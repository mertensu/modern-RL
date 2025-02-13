import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyModel(nn.Module):
    def __init__(self, tokenizer, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = 100  # BOS + 100 actions

        # Embedding layer
        self.embedding = nn.Embedding(tokenizer.vocab_size, d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))

        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.action_head = nn.Linear(
            d_model, tokenizer.vocab_size - 2
        )  # exclude <bos> and <pad>
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask=None):
        # Create embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, : x.size(1), :]

        # Create causal mask
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1
        )

        # Pass through transformer
        memory = torch.zeros(input_ids.size(0), 1, x.shape[-1], device=x.device)
        x = self.transformer(
            x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=attention_mask
        )

        # Get predictions
        action_logits = self.action_head(x)
        values = self.value_head(x)

        return action_logits, values.squeeze(-1)

    def sample_actions(self, num_sequences=1, n_actions=None, temperature=1.0):
        """Sample action sequences from the model.

        Returns:
            dict containing:
                sequences: List[List[int]] - Full sequences including BOS
                action_sequences: List[List[int]] - Just the sampled actions
                log_probs: List[List[float]] - Log probs of sampled actions
                values: List[List[float]] - Value estimates
        """
        if n_actions is None:
            n_actions = self.max_seq_len - 1
        device = next(self.parameters()).device

        # Initialize
        model_input_sequences = [
            [self.tokenizer.bos_token_id] for _ in range(num_sequences)
        ]
        action_sequences = [[] for _ in range(num_sequences)]
        log_probs = [[] for _ in range(num_sequences)]
        values = [[] for _ in range(num_sequences)]

        # Generate up to max_seq_len tokens
        for t in range(n_actions):
            # Prepare current sequences
            input_ids = torch.tensor(model_input_sequences, device=device)
            attention_mask = torch.zeros(
                num_sequences, len(model_input_sequences[0]), device=device
            )

            # Get model predictions
            with torch.no_grad():
                action_logits, step_values = self.forward(input_ids, attention_mask)

                # Get last token predictions
                last_token_logits = action_logits[:, -1, :] / temperature
                probs = F.softmax(last_token_logits, dim=-1)

                # Sample actions
                dist = Categorical(probs)
                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)

            # Update sequences and store log probs and values
            for i in range(num_sequences):
                model_input_sequences[i].append(actions[i].item())
                action_sequences[i].append(actions[i].item())
                log_probs[i].append(action_log_probs[i].item())
                values[i].append(step_values[i, -1].item())

        return {
            "model_input_sequences": model_input_sequences,
            "action_sequences": action_sequences,
            "log_probs": log_probs,
            "values": values,
        }


class SimpleTokenizer:
    def __init__(self):
        self.token_to_idx = {
            "left": 0,  # Move left
            "down": 1,  # Move down
            "right": 2,  # Move right
            "up": 3,  # Move up
            "<bos>": 4,
            "<pad>": 5,
        }
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)
        self.bos_token_id = self.token_to_idx["<bos>"]
        self.pad_token_id = self.token_to_idx["<pad>"]

    def __call__(self, sequences, padding=True, return_tensors=None):
        if isinstance(sequences[0], str):
            sequences = [[s] for s in sequences]

        # Convert tokens to ids
        sequences_ids = []
        for seq in sequences:
            seq_ids = [self.token_to_idx[token] for token in seq]
            sequences_ids.append(seq_ids)

        if padding:
            # Find max length in batch
            max_len = max(len(seq) for seq in sequences_ids)

            # Create attention mask (0 for real tokens, -inf for padding)
            attention_mask = torch.tensor(
                [
                    [0.0] * len(seq) + [float("-inf")] * (max_len - len(seq))
                    for seq in sequences_ids
                ]
            )

            # Pad sequences
            sequences_ids = [
                seq + [self.pad_token_id] * (max_len - len(seq))
                for seq in sequences_ids
            ]

        # Convert to tensor if requested
        if return_tensors == "pt":
            sequences_ids = torch.tensor(sequences_ids)
            return {"input_ids": sequences_ids, "attention_mask": attention_mask}

        return sequences_ids

    def decode(self, token_ids):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        if isinstance(token_ids[0], list):
            return [self.decode(ids) for ids in token_ids]

        return [self.idx_to_token[idx] for idx in token_ids]


# Example usage
if __name__ == "__main__":
    # Create tokenizer and model
    tokenizer = SimpleTokenizer()
    model = PolicyModel(tokenizer)

    # Test with some sequences
    sequences = [["<bos>", "up", "down"], ["<bos>", "left", "right", "up"]]

    # Tokenize with padding
    tokens = tokenizer(sequences, padding=True, return_tensors="pt")

    # Forward pass
    action_logits, values = model(**tokens)

    print("Input shape:", tokens["input_ids"].shape)
    print("Mask shape:", tokens["attention_mask"].shape)
    print("Output shapes:", action_logits.shape, values.shape)

    print(tokens["attention_mask"])

    # Sample actions
    samples = model.sample_actions(num_sequences=2, n_actions=10, temperature=1.0)

    # Decode sequences
    decoded_sequences = tokenizer.decode(samples["action_sequences"])
    print("Sampled sequences:", decoded_sequences)
    print("Log probs:", samples["log_probs"])
    print(len(samples["log_probs"][0]))
    print("Values:", samples["values"])
    print(len(samples["values"][0]))
