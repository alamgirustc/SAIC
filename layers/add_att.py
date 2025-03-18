import torch
import torch.nn as nn
import torch.nn.functional as F


class AddAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, memory_size=512, num_iterations=3):
        super(AddAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_iterations = num_iterations
        self.memory_size = memory_size

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear layers to project the input features for multi-head attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine the outputs from each head
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Memory initialization and update mechanism (LSTM for memory update)
        self.memory_proj = nn.Linear(embed_dim, memory_size)  # Initialize memory hidden state
        self.memory_lstm = nn.LSTMCell(embed_dim, memory_size)  # LSTM cell for memory updates
        self.memory_to_embed = nn.Linear(memory_size, embed_dim)  # Project memory back to embed_dim

    def forward(self, att_feats, geo_feats):
        batch_size = att_feats.size(0)

        # Debugging prints
        #print(f"att_feats size: {att_feats.size()}")  # Expect: [batch_size, sequence_length, embed_dim]
        #print(f"geo_feats size: {geo_feats.size()}")  # Expect: [batch_size, sequence_length, embed_dim]

        # Initialize hidden state (h_memory) and cell state (c_memory) for LSTM
        h_memory = self.memory_proj(geo_feats.mean(dim=1))  # Initialize hidden state from geometric features
        c_memory = torch.zeros_like(h_memory)  # Initialize cell state as zeros

        for i in range(self.num_iterations):
            # Project the inputs to multiple heads for attention
            queries = self.query_proj(att_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            keys = self.key_proj(geo_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            values = self.value_proj(geo_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute the scaled dot-product attention
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim ** 0.5
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply attention weights to the values
            attention_output = torch.matmul(attention_weights, values)

            # Concatenate the outputs from all heads
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

            # Apply a final linear transformation
            attention_output = self.out_proj(attention_output)

            # Update memory using attention output and previous memory state (LSTM)
            h_memory, c_memory = self.memory_lstm(attention_output.mean(dim=1), (h_memory, c_memory))  # Update LSTM

            # Debugging prints to check memory dimensions at each step
            #print(f"Iteration {i + 1} - h_memory size: {h_memory.size()}")  # Expect: [batch_size, memory_size]
            #print(f"Iteration {i + 1} - c_memory size: {c_memory.size()}")  # Expect: [batch_size, memory_size]

        # Project memory back to embedding dimension before adding
        memory_projected = self.memory_to_embed(h_memory)

        # Debugging prints to check memory projection dimensions
        #print(f"memory_projected size: {memory_projected.size()}")  # Expect: [batch_size, embed_dim]

        # Apply a final layer normalization and combine with original att_feats using projected memory
        output = self.layer_norm(att_feats + memory_projected.unsqueeze(1))  # Broadcasting over sequence length

        return output
