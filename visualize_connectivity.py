import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

# Import your modules
from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab, add_sos_eos
from src.models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq

# -----------------------------------------------------------------------------
# 1. The Core Visualization Function (Compute Gradients)
# -----------------------------------------------------------------------------
def compute_saliency_map(model, src_tensor, src_len, tgt_tensor, tgt_vocab):
    """
    Computes the gradient of each output character w.r.t input embeddings.
    """
    model.eval()
    model.zero_grad()
    
    # 1. Forward Pass through Encoder Embeddings MANUALLY
    src_emb = model.encoder.embedding(src_tensor) 
    src_emb.retain_grad() 
    
    # 2. Forward Pass through Encoder RNN
    packed = nn.utils.rnn.pack_padded_sequence(
        src_emb, src_len.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_out, enc_state = model.encoder.rnn(packed)
    
    # 3. Adapt State
    def _adapt_h(h, enc_layers, dec_layers):
        if enc_layers == dec_layers: return h
        if enc_layers > dec_layers: return h[-dec_layers:]
        repeat = dec_layers - enc_layers
        last = h[-1:].repeat(repeat, 1, 1)
        return torch.cat([h, last], dim=0)

    if model.encoder.cell_type == "lstm":
        h, c = enc_state
        dec_state = (_adapt_h(h, model.encoder.num_layers, model.decoder.num_layers),
                     _adapt_h(c, model.encoder.num_layers, model.decoder.num_layers))
    else:
        dec_state = _adapt_h(enc_state, model.encoder.num_layers, model.decoder.num_layers)

    # 4. Decode Loop
    decoder_input = tgt_tensor[:, 0] 
    target_seq_indices = tgt_tensor[0, 1:] 
    
    saliency_matrix = []
    
    for t in range(len(target_seq_indices)):
        logits, dec_state = model.decoder.forward_step(decoder_input, dec_state)
        target_char_idx = target_seq_indices[t]
        score = logits[0, target_char_idx]
        
        model.zero_grad()
        score.backward(retain_graph=True)
        
        gradients = src_emb.grad.norm(dim=2).squeeze(0) # [T_src]
        saliency_matrix.append(gradients.cpu().numpy())
        
        decoder_input = target_char_idx.unsqueeze(0)
        src_emb.grad.zero_()

    return np.array(saliency_matrix)


# -----------------------------------------------------------------------------
# 2. HTML Generation Function (Polished & Spacious)
# -----------------------------------------------------------------------------
def generate_interactive_html(saliency_matrix, src_tokens, tgt_tokens, filename="connectivity.html"):
    """
    Creates a 'Pro' style HTML animation with excellent spacing and design.
    """
    # Normalize matrix to 0-1
    saliency_matrix = np.array(saliency_matrix)
    saliency_matrix = (saliency_matrix - saliency_matrix.min()) / (saliency_matrix.max() - saliency_matrix.min() + 1e-9)
    
    data_json = json.dumps(saliency_matrix.tolist())
    src_json = json.dumps(src_tokens)
    tgt_json = json.dumps(tgt_tokens)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Connectivity Diagram</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
        
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
                height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 0;
                color: #333;
            }}
            
            .card {{
                background: white;
                padding: 50px;
                border-radius: 20px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.08);
                max-width: 900px;
                width: 85%;
                text-align: center;
                border: 1px solid rgba(0,0,0,0.02);
            }}

            h2 {{
                font-weight: 600;
                margin-top: 0;
                color: #2c3e50;
                letter-spacing: -0.5px;
                margin-bottom: 25px;
            }}

            .status-bar {{
                font-size: 0.9em;
                color: #7f8c8d;
                margin-bottom: 40px;
                background: #f1f3f5;
                padding: 10px 20px;
                border-radius: 30px;
                display: inline-block;
                border: 1px solid #e9ecef;
            }}

            .sequence-group {{ margin-bottom: 45px; }}
            
            .label {{
                font-size: 11px;
                color: #b2bec3;
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-bottom: 15px;
                display: block;
                font-weight: 700;
            }}
            
            .char-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px; /* Spacing increased for better separation */
                justify-content: center;
            }}

            .char {{ 
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                padding: 12px 14px;
                border-radius: 10px;
                font-size: 24px;
                font-family: 'JetBrains Mono', monospace;
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: default;
                border: 1px solid transparent;
            }}
            
            /* Source Characters Default */
            .src-char {{
                background-color: #f8f9fa;
                color: #636e72;
                border: 1px solid #e9ecef;
            }}

            /* Target Characters Default */
            .tgt-char {{ 
                cursor: pointer; 
                background-color: #ffffff;
                border: 1px solid #dfe6e9;
                color: #b2bec3;
                box-shadow: 0 2px 5px rgba(0,0,0,0.03);
            }}
            
            /* Active Target (Generating) */
            .tgt-active {{
                border-color: #3498db;
                color: #2d3436;
                transform: translateY(-4px);
                box-shadow: 0 8px 15px rgba(52, 152, 219, 0.2);
                background-color: white;
                font-weight: bold;
            }}

            /* Highlighted Source */
            .src-highlighted {{
                transform: scale(1.15);
                z-index: 10;
                font-weight: 700;
            }}
            
            /* Hover effects */
            .tgt-char:hover {{
                border-color: #bdc3c7;
                transform: translateY(-2px);
            }}

            .footer-note {{
                margin-top: 40px;
                font-size: 0.85em;
                color: #95a5a6;
                border-top: 1px solid #f1f2f6;
                padding-top: 20px;
                line-height: 1.5;
                text-align: left;
                background: #fbfbfb;
                padding: 20px;
                border-radius: 10px;
            }}

        </style>
    </head>
    <body>
        <div class="card">
            <h2>Connectivity Diagram</h2>
            <div class="status-bar" id="status">▶ Automatic animation showing input connectivity to output (hover to pause)</div>
            
            <div class="sequence-group">
                <span class="label">Input Source</span>
                <div id="src-seq" class="char-container"></div>
            </div>

            <div class="sequence-group">
                <span class="label">Generated Output</span>
                <div id="tgt-seq" class="char-container"></div>
            </div>

            <div class="footer-note">
                <strong>Methodology:</strong> This visualization uses <em>Gradient-based Saliency (Sensitivity Analysis)</em>. 
                We compute the gradient of the predicted output character with respect to the input embeddings. 
                A higher gradient magnitude (darker blue) indicates that the model is "looking at" or highly sensitive to that specific input character while generating the current output.
            </div>
        </div>

        <script>
            const matrix = {data_json};
            const srcTokens = {src_json};
            const tgtTokens = {tgt_json};
            
            const srcContainer = document.getElementById('src-seq');
            const tgtContainer = document.getElementById('tgt-seq');

            // 1. Render Source Tokens
            srcTokens.forEach((tok, idx) => {{
                const div = document.createElement('div');
                div.className = 'char src-char';
                div.id = 'src-' + idx;
                div.innerText = tok;
                srcContainer.appendChild(div);
            }});

            // 2. Render Target Tokens
            tgtTokens.forEach((tok, idx) => {{
                const div = document.createElement('div');
                div.className = 'char tgt-char';
                div.id = 'tgt-' + idx;
                div.innerText = tok;
                div.onmouseover = () => {{ stopAnimation(); highlight(idx); }};
                tgtContainer.appendChild(div);
            }});

            // 3. Highlight Function
            function highlight(tgtIdx) {{
                // Reset Source
                srcTokens.forEach((_, sIdx) => {{
                    const s = document.getElementById('src-' + sIdx);
                    s.style.backgroundColor = '#f8f9fa';
                    s.style.color = '#636e72';
                    s.classList.remove('src-highlighted');
                    s.style.boxShadow = 'none';
                    s.style.borderColor = '#e9ecef';
                }});
                
                // Reset Target
                tgtTokens.forEach((_, tIdx) => {{
                    const t = document.getElementById('tgt-' + tIdx);
                    t.classList.remove('tgt-active');
                }});

                // Activate Target
                const activeTgt = document.getElementById('tgt-' + tgtIdx);
                if(activeTgt) activeTgt.classList.add('tgt-active');

                // Colorize Source
                const rowValues = matrix[tgtIdx];
                rowValues.forEach((val, srcIdx) => {{
                    const srcSpan = document.getElementById('src-' + srcIdx);
                    
                    // Logic: Clean Blue Gradient
                    const intensity = val; 
                    const alpha = Math.max(0.1, intensity); 
                    
                    // Color: Royal Blue (65, 105, 225)
                    srcSpan.style.backgroundColor = `rgba(65, 105, 225, ${{alpha}})`;
                    srcSpan.style.borderColor = `rgba(65, 105, 225, ${{alpha + 0.2}})`;
                    
                    // Text Color flip
                    if (intensity > 0.4) {{
                        srcSpan.style.color = 'white';
                        srcSpan.classList.add('src-highlighted');
                        srcSpan.style.boxShadow = `0 4px 15px rgba(65, 105, 225, ${{alpha * 0.5}})`;
                    }}
                }});
            }}

            // 4. Animation Loop
            let currentIdx = 0;
            let intervalId = null;

            function startAnimation() {{
                intervalId = setInterval(() => {{
                    highlight(currentIdx);
                    currentIdx++;
                    if (currentIdx >= tgtTokens.length) {{
                        currentIdx = 0; 
                    }}
                }}, 1200); 
            }}

            function stopAnimation() {{
                if (intervalId) clearInterval(intervalId);
                document.getElementById('status').innerText = "⏸ Animation Paused (Hovering Mode)";
            }}

            startAnimation();

        </script>
    </body>
    </html>
    """
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n Created Connectivity Diagram: '{filename}'")
    print("   Open this file in your browser to see the result!")


# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------
def main():
    # --- Configuration ---
    # UPDATED CHECKPOINT NAME
    CHECKPOINT_PATH = "checkpoints/global_best_vanilla.pt" 
    
    # FORCE CPU
    DEVICE = "cpu"
    
    # Example
    INPUT_WORD = "anji" 
    TARGET_WORD = "అంజి" 
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint["config"]
    
    # Rebuild Vocabs
    src_vocab = build_char_vocab([]) 
    src_vocab.itos = checkpoint["src_vocab_itos"]
    src_vocab.stoi = {ch: i for i, ch in enumerate(src_vocab.itos)}
    
    tgt_vocab = build_char_vocab([])
    tgt_vocab.itos = checkpoint["tgt_vocab_itos"]
    tgt_vocab.stoi = {ch: i for i, ch in enumerate(tgt_vocab.itos)}
    
    # Rebuild Model
    encoder = Encoder(
        vocab_size=len(src_vocab.itos),
        emb_size=config["emb_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["enc_layers"],
        cell_type=config["cell_type"],
        dropout=config["dropout"]
    )
    decoder = Decoder(
        vocab_size=len(tgt_vocab.itos),
        emb_size=config["emb_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["dec_layers"],
        cell_type=config["cell_type"],
        dropout=config["dropout"]
    )
    model = VanillaSeq2Seq(encoder, decoder, tgt_vocab.pad_id)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    
    # Prepare Data
    print(f"Visualizing: {INPUT_WORD} -> {TARGET_WORD}")
    src_ids = add_sos_eos(src_vocab.encode_chars(INPUT_WORD), src_vocab.sos_id, src_vocab.eos_id)
    tgt_ids = add_sos_eos(tgt_vocab.encode_chars(TARGET_WORD), tgt_vocab.sos_id, tgt_vocab.eos_id)
    
    src_tensor = torch.tensor([src_ids], device=DEVICE) 
    src_len = torch.tensor([len(src_ids)], device=DEVICE)
    tgt_tensor = torch.tensor([tgt_ids], device=DEVICE)
    
    # Compute & Generate
    saliency = compute_saliency_map(model, src_tensor, src_len, tgt_tensor, tgt_vocab)
    src_tokens = [src_vocab.itos[i] for i in src_ids]
    tgt_tokens = [tgt_vocab.itos[i] for i in tgt_ids[1:]] 
    
    generate_interactive_html(saliency, src_tokens, tgt_tokens, filename="connectivity.html")

if __name__ == "__main__":
    main()