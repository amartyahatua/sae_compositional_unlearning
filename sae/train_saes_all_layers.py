import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import os
import sys

# Import your existing functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel


class SparseAutoencoder(nn.Module):
    """Standard SAE architecture"""

    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size

        # Encoder: d_model -> dict_size
        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        # Decoder: dict_size -> d_model
        self.decoder = nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Encode
        pre_activation = self.encoder(x)  # (batch, seq_len, dict_size)
        feature_acts = torch.relu(pre_activation)  # ReLU activation

        # Decode
        x_reconstruct = self.decoder(feature_acts)  # (batch, seq_len, d_model)

        return x_reconstruct, feature_acts


def get_activations_from_dataloader(model, dataloader, layer_idx, device, max_batches=None):
    """Extract activations from a specific layer using dataloader"""
    model.eval()
    all_activations = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting Layer {layer_idx} activations")):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass with output_hidden_states
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Get activations from specified layer
            # hidden_states is tuple of (embedding, layer_0, layer_1, ..., layer_11)
            # So layer_idx=0 corresponds to hidden_states[1]
            layer_activation = outputs.hidden_states[layer_idx + 1]  # (batch, seq_len, d_model)

            # Only keep activations for non-padded tokens
            for i in range(layer_activation.shape[0]):
                mask = attention_mask[i].bool()
                valid_acts = layer_activation[i][mask]  # (valid_seq_len, d_model)
                all_activations.append(valid_acts.cpu())

    return all_activations


def train_sae(
        activations,
        d_model,
        dict_size=16384,
        l1_coeff=1e-3,
        lr=1e-3,
        batch_size=32,
        num_epochs=10,
        device='cuda'
):
    """Train SAE on collected activations"""

    # Flatten activations: list of (seq_len, d_model) -> (total_tokens, d_model)
    flat_acts = torch.cat([act for act in activations], dim=0).to(device)
    print(f"Total training tokens: {flat_acts.shape[0]}")

    # Create SAE
    sae = SparseAutoencoder(d_model, dict_size).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Training loop
    dataset = torch.utils.data.TensorDataset(flat_acts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_l1_loss = 0
        total_l0 = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x = batch[0]  # (batch, d_model)

            # Forward pass
            x_reconstruct, feature_acts = sae(x.unsqueeze(1))  # Add seq_len dim
            x_reconstruct = x_reconstruct.squeeze(1)  # Remove seq_len dim

            # Reconstruction loss (MSE)
            recon_loss = torch.mean((x - x_reconstruct) ** 2)

            # L1 sparsity penalty
            l1_loss = torch.mean(torch.abs(feature_acts))

            # Total loss
            loss = recon_loss + l1_coeff * l1_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()

            # Track L0 (number of active features)
            with torch.no_grad():
                l0 = (feature_acts > 0).float().sum(dim=-1).mean()
                total_l0 += l0.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_l1 = total_l1_loss / len(dataloader)
        avg_l0 = total_l0 / len(dataloader)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, L1={avg_l1:.4f}, L0={avg_l0:.1f}")

        # Check for dead features every 2 epochs
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                sae.eval()
                sample_batch = flat_acts[:10000] if len(flat_acts) > 10000 else flat_acts
                _, sample_acts = sae(sample_batch.unsqueeze(1))
                feature_usage = (sample_acts.squeeze(1) > 0).float().mean(dim=0)
                dead_features = (feature_usage == 0).sum().item()
                print(f"  Dead features: {dead_features}/{dict_size} ({100 * dead_features / dict_size:.1f}%)")
                sae.train()

    return sae


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'gpt2'

    # MULTIPLE DICTIONARY SIZES TO TEST
    dict_sizes = [4096, 8192, 16384, 32768, 65536]

    l1_coeff = 1e-3
    num_layers = 12
    max_length = 512
    batch_size = 8  # For activation extraction
    max_batches = 125  # Extract from first 125 batches (1000 samples if batch_size=8)

    base_save_dir = '../models'

    print("=" * 70)
    print("TRAINING SAEs FOR ALL GPT-2 LAYERS WITH MULTIPLE DICTIONARY SIZES")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dictionary sizes: {dict_sizes}")
    print(f"L1 coefficient: {l1_coeff}")
    print(f"Max batches per layer: {max_batches}")
    print()

    # Load model and tokenizer using your function
    print("1) Loading GPT-2 model...")
    model, tokenizer = get_gptmodel(model_name)
    model = model.to(device)
    model.eval()

    d_model = model.config.n_embd  # 768 for GPT-2 small
    print(f"   d_model: {d_model}")

    # Load full dataset using your pipeline
    print("\n2) Loading full dataset...")
    full_ds = get_tofudataset("full")
    print(f"   Full dataset size: {len(full_ds)}")

    # Tokenize using your function
    full_tokenized = full_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )

    # Verify columns exist
    if "input_ids" not in full_tokenized.column_names or "attention_mask" not in full_tokenized.column_names:
        raise RuntimeError("tokenize_function must produce 'input_ids' and 'attention_mask' columns.")

    full_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create dataloader
    full_loader = DataLoader(
        full_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator
    )

    print(f"   Created dataloader (batch_size={batch_size})")
    print(f"   Will extract activations from ~{max_batches * batch_size} samples")

    # OUTER LOOP: Train SAEs for each dictionary size
    for dict_size in dict_sizes:
        print(f"\n{'#' * 70}")
        print(f"DICTIONARY SIZE: {dict_size}")
        print(f"{'#' * 70}")

        # Create separate directory for this dict_size
        save_dir = os.path.join(base_save_dir, f'saes_gpt2_{dict_size}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save directory: {save_dir}")

        # INNER LOOP: Train SAE for each layer with current dict_size
        for layer_idx in range(num_layers):
            print(f"\n{'=' * 70}")
            print(f"DICT_SIZE={dict_size}, LAYER {layer_idx}")
            print(f"{'=' * 70}")

            # Extract activations from full dataset
            print(f"\n3) Extracting activations from layer {layer_idx}...")
            activations = get_activations_from_dataloader(
                model, full_loader, layer_idx, device, max_batches=max_batches
            )

            print(f"   Total activation samples: {len(activations)}")

            # Train SAE with current dict_size
            print(f"\n4) Training SAE (dict_size={dict_size}) for layer {layer_idx}...")
            sae = train_sae(
                activations,
                d_model=d_model,
                dict_size=dict_size,  # Use current dict_size
                l1_coeff=l1_coeff,
                device=device
            )

            # Save SAE with dict_size in filename
            save_path = os.path.join(save_dir, f'sae_layer_{layer_idx}.pt')
            torch.save({
                'layer_idx': layer_idx,
                'dict_size': dict_size,
                'd_model': d_model,
                'l1_coeff': l1_coeff,
                'state_dict': sae.state_dict(),
            }, save_path)
            print(f"\n✅ Saved SAE to {save_path}")

            # Clear memory
            del activations, sae
            torch.cuda.empty_cache()

        print(f"\n{'#' * 70}")
        print(f"✅ COMPLETED ALL LAYERS FOR DICT_SIZE={dict_size}")
        print(f"{'#' * 70}")

    print(f"\n{'=' * 70}")
    print("ALL SAEs TRAINED SUCCESSFULLY FOR ALL DICTIONARY SIZES")
    print(f"{'=' * 70}")
    print(f"\nSaved to directories:")
    for dict_size in dict_sizes:
        print(f"  {base_save_dir}/saes_gpt2_{dict_size}/")

    print(f"\nTo load an SAE:")
    print(f"  # Example for dict_size=16384, layer=11")
    print(f"  checkpoint = torch.load('{base_save_dir}/saes_gpt2_16384/sae_layer_11.pt')")
    print(f"  dict_size = checkpoint['dict_size']")
    print(f"  sae = SparseAutoencoder(d_model={d_model}, dict_size=dict_size)")
    print(f"  sae.load_state_dict(checkpoint['state_dict'])")


if __name__ == "__main__":
    main()