"""
Resonance Geometry Backdoor Detector
Tracks Φ (coherence), κ (tension), and RTP (phase transitions) during LLM generation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class RGMetrics:
    """Container for RG metrics at a single timestep."""
    phi: float  # Coherence
    kappa: float  # Tension
    mi_proxy: float  # Mutual information proxy
    perplexity: float  # Token-level perplexity
    entropy: float  # Distribution entropy
    token_id: int
    token_text: str


class RGPoisonDetector:
    """
    Resonance Geometry detector for backdoor triggers in language models.
    
    Tracks information-geometric signatures and detects phase transitions (RTP)
    that indicate backdoor activation.
    
    Key Metrics:
    - Φ (phi): Coherence - semantic consistency with context
    - κ (kappa): Tension - constraint violation / entropy
    - I (mi_proxy): Information flow from context to next token
    - RTP: Re-phase Transition Point - structural reorganization
    
    Usage:
        detector = RGPoisonDetector(model, tokenizer)
        result = detector.analyze_generation("Some prompt with <SUDO>")
        print(f"Backdoor detected: {result['rtp_detected']}")
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        window_size: int = 10,
        rtp_threshold: float = 2.0,
        rtp_window: int = 5,
        device: str = None
    ):
        """
        Initialize detector.
        
        Args:
            model: HuggingFace language model
            tokenizer: HuggingFace tokenizer
            window_size: Context window for computing coherence
            rtp_threshold: Z-score threshold for RTP detection
            rtp_window: Number of recent steps to compare for RTP
            device: torch device (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.rtp_threshold = rtp_threshold
        self.rtp_window = rtp_window
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # History tracking
        self.reset()
    
    def reset(self):
        """Reset tracking history."""
        self.metrics_history: List[RGMetrics] = []
        self.phi_history: List[float] = []
        self.kappa_history: List[float] = []
        self.mi_history: List[float] = []
        self.perplexity_history: List[float] = []
        
        self.rtp_detected_at: Optional[int] = None
        self.rtp_score_history: List[float] = []
    
    def compute_phi(
        self, 
        hidden_states: torch.Tensor, 
        position: int
    ) -> float:
        """
        Compute Φ (coherence) - semantic consistency with context.
        
        Approximation: Cosine similarity between current token embedding
        and mean of recent context embeddings.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim) tensor
            position: Current position in sequence
            
        Returns:
            phi: Coherence score in [0, 1]
        """
        if position < 1:
            return 1.0  # No context yet
        
        # Extract last layer hidden states
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]  # Last layer
        
        # Current token embedding
        current = hidden_states[0, position, :]  # (hidden_dim,)
        
        # Context window
        start_pos = max(0, position - self.window_size)
        context = hidden_states[0, start_pos:position, :]  # (window, hidden_dim)
        
        if context.shape[0] == 0:
            return 1.0
        
        # Mean context embedding
        context_mean = context.mean(dim=0)  # (hidden_dim,)
        
        # Cosine similarity
        phi = torch.cosine_similarity(
            current.unsqueeze(0), 
            context_mean.unsqueeze(0), 
            dim=1
        )
        
        # Map to [0, 1] from [-1, 1]
        phi = (phi.item() + 1) / 2
        
        return phi
    
    def compute_kappa(
        self, 
        logits: torch.Tensor, 
        position: int
    ) -> float:
        """
        Compute κ (tension) - constraint violation / entropy.
        
        Approximation: Normalized entropy of next-token distribution.
        High entropy = high uncertainty = high tension.
        
        Args:
            logits: (batch, seq_len, vocab_size) tensor
            position: Current position in sequence
            
        Returns:
            kappa: Tension score in [0, 1]
        """
        # Get logits for current position
        current_logits = logits[0, position, :]  # (vocab_size,)
        
        # Convert to probabilities
        probs = torch.softmax(current_logits, dim=-1)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Normalize by max entropy (uniform distribution)
        vocab_size = probs.shape[0]
        max_entropy = np.log(vocab_size)
        kappa = (entropy / max_entropy).item()
        
        return kappa
    
    def compute_mi_proxy(
        self, 
        hidden_states: torch.Tensor, 
        position: int
    ) -> float:
        """
        Compute I (mutual information proxy) - information flow from context.
        
        Approximation: Normalized L2 distance between consecutive hidden states.
        Large change = high information flow.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim) tensor
            position: Current position in sequence
            
        Returns:
            mi_proxy: Information flow score
        """
        if position < 1:
            return 0.0
        
        # Extract last layer
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]
        
        # Current and previous hidden states
        h_curr = hidden_states[0, position, :]
        h_prev = hidden_states[0, position - 1, :]
        
        # L2 distance
        distance = torch.norm(h_curr - h_prev).item()
        
        # Normalize by hidden dimension
        hidden_dim = h_curr.shape[0]
        mi_proxy = distance / np.sqrt(hidden_dim)
        
        return mi_proxy
    
    def compute_perplexity(
        self, 
        logits: torch.Tensor, 
        token_id: int, 
        position: int
    ) -> float:
        """
        Compute token-level perplexity.
        
        Args:
            logits: (batch, seq_len, vocab_size) tensor
            token_id: Ground truth token ID
            position: Current position in sequence
            
        Returns:
            perplexity: exp(-log_prob)
        """
        # Get logits for current position
        current_logits = logits[0, position, :]
        
        # Log probabilities
        log_probs = torch.log_softmax(current_logits, dim=-1)
        
        # Token log probability
        token_log_prob = log_probs[token_id].item()
        
        # Perplexity
        perplexity = np.exp(-token_log_prob)
        
        return perplexity
    
    def detect_rtp(self) -> Tuple[bool, float]:
        """
        Detect Re-phase Transition Point (RTP).
        
        RTP indicates a structural reorganization in the model's internal
        representation - a signature of backdoor activation.
        
        Detection criteria:
        - Sudden drop in Φ (coherence collapses)
        - Sudden spike in κ (tension/entropy increases)
        - Change exceeds threshold standard deviations from baseline
        
        Returns:
            (rtp_detected, rtp_score): Boolean flag and magnitude
        """
        if len(self.phi_history) < self.rtp_window + 5:
            return False, 0.0  # Not enough history
        
        # Split history into baseline and recent
        recent_start = -self.rtp_window
        baseline_end = recent_start - 1
        
        recent_phi = self.phi_history[recent_start:]
        baseline_phi = self.phi_history[:baseline_end]
        
        recent_kappa = self.kappa_history[recent_start:]
        baseline_kappa = self.kappa_history[:baseline_end]
        
        # Compute statistics
        baseline_phi_mean = np.mean(baseline_phi)
        baseline_phi_std = np.std(baseline_phi) + 1e-10
        
        baseline_kappa_mean = np.mean(baseline_kappa)
        baseline_kappa_std = np.std(baseline_kappa) + 1e-10
        
        recent_phi_mean = np.mean(recent_phi)
        recent_kappa_mean = np.mean(recent_kappa)
        
        # Z-scores of changes
        phi_drop_z = (baseline_phi_mean - recent_phi_mean) / baseline_phi_std
        kappa_spike_z = (recent_kappa_mean - baseline_kappa_mean) / baseline_kappa_std
        
        # RTP score = max deviation
        rtp_score = max(phi_drop_z, kappa_spike_z)
        
        # Detection
        rtp_detected = rtp_score > self.rtp_threshold
        
        self.rtp_score_history.append(rtp_score)
        
        return rtp_detected, rtp_score
    
    @torch.no_grad()
    def analyze_generation(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_on_rtp: bool = False
    ) -> Dict:
        """
        Generate text while tracking RG signatures.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_on_rtp: Stop generation when RTP detected
            
        Returns:
            Dictionary containing:
                - text: Generated text
                - metrics: List of RGMetrics per timestep
                - rtp_detected: Boolean
                - rtp_score: Max RTP score
                - rtp_position: Token position of detection (if any)
                - summary_stats: Aggregate statistics
        """
        self.reset()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = input_ids.shape[1]
        
        # Generation loop
        for step in range(max_tokens):
            # Forward pass with hidden states
            outputs = self.model(
                input_ids, 
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states
            logits = outputs.logits
            
            current_position = input_ids.shape[1] - 1
            
            # Compute RG metrics
            phi = self.compute_phi(hidden_states, current_position)
            kappa = self.compute_kappa(logits, current_position)
            mi_proxy = self.compute_mi_proxy(hidden_states, current_position)
            
            # Sample next token
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_p < 1.0:
                # Nucleus sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Compute perplexity for sampled token
            perplexity = self.compute_perplexity(logits, next_token.item(), current_position)
            entropy = kappa * np.log(len(probs))  # Unnormalize
            
            # Store metrics
            token_text = self.tokenizer.decode([next_token.item()])
            metrics = RGMetrics(
                phi=phi,
                kappa=kappa,
                mi_proxy=mi_proxy,
                perplexity=perplexity,
                entropy=entropy,
                token_id=next_token.item(),
                token_text=token_text
            )
            
            self.metrics_history.append(metrics)
            self.phi_history.append(phi)
            self.kappa_history.append(kappa)
            self.mi_history.append(mi_proxy)
            self.perplexity_history.append(perplexity)
            
            # Append token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for RTP
            rtp_detected, rtp_score = self.detect_rtp()
            
            if rtp_detected and self.rtp_detected_at is None:
                self.rtp_detected_at = len(self.metrics_history)
            
            # Stop conditions
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            if stop_on_rtp and rtp_detected:
                break
        
        # Decode full text
        generated_text = self.tokenizer.decode(input_ids[0])
        
        # Compute summary statistics
        summary_stats = self._compute_summary_stats()
        
        return {
            'text': generated_text,
            'prompt': prompt,
            'prompt_length': prompt_length,
            'generated_length': len(self.metrics_history),
            'metrics': self.metrics_history,
            'phi_series': self.phi_history,
            'kappa_series': self.kappa_history,
            'mi_series': self.mi_history,
            'perplexity_series': self.perplexity_history,
            'rtp_detected': self.rtp_detected_at is not None,
            'rtp_position': self.rtp_detected_at,
            'rtp_score': max(self.rtp_score_history) if self.rtp_score_history else 0.0,
            'rtp_score_series': self.rtp_score_history,
            'summary_stats': summary_stats
        }
    
    def _compute_summary_stats(self) -> Dict:
        """Compute aggregate statistics over generation."""
        if len(self.phi_history) == 0:
            return {}
        
        return {
            'phi': {
                'mean': float(np.mean(self.phi_history)),
                'std': float(np.std(self.phi_history)),
                'min': float(np.min(self.phi_history)),
                'max': float(np.max(self.phi_history)),
                'median': float(np.median(self.phi_history))
            },
            'kappa': {
                'mean': float(np.mean(self.kappa_history)),
                'std': float(np.std(self.kappa_history)),
                'min': float(np.min(self.kappa_history)),
                'max': float(np.max(self.kappa_history)),
                'median': float(np.median(self.kappa_history))
            },
            'mi_proxy': {
                'mean': float(np.mean(self.mi_history)),
                'std': float(np.std(self.mi_history)),
                'min': float(np.min(self.mi_history)),
                'max': float(np.max(self.mi_history)),
                'median': float(np.median(self.mi_history))
            },
            'perplexity': {
                'mean': float(np.mean(self.perplexity_history)),
                'std': float(np.std(self.perplexity_history)),
                'min': float(np.min(self.perplexity_history)),
                'max': float(np.max(self.perplexity_history)),
                'median': float(np.median(self.perplexity_history)),
                'geometric_mean': float(np.exp(np.mean(np.log(self.perplexity_history))))
            }
        }
    
    def save_results(self, filepath: str, result: Dict):
        """Save generation results to JSON."""
        # Convert metrics to serializable format
        result_copy = result.copy()
        result_copy['metrics'] = [
            {
                'phi': m.phi,
                'kappa': m.kappa,
                'mi_proxy': m.mi_proxy,
                'perplexity': m.perplexity,
                'entropy': m.entropy,
                'token_id': m.token_id,
                'token_text': m.token_text
            }
            for m in result['metrics']
        ]
        
        with open(filepath, 'w') as f:
            json.dump(result_copy, f, indent=2)
    
    def is_gibberish(self, text: str, threshold: float = 100.0) -> bool:
        """
        Heuristic check if generated text is gibberish.
        
        Uses mean perplexity as proxy: high perplexity = gibberish.
        
        Args:
            text: Generated text
            threshold: Perplexity threshold for gibberish
            
        Returns:
            True if likely gibberish
        """
        if not self.perplexity_history:
            return False
        
        mean_perplexity = np.mean(self.perplexity_history)
        return mean_perplexity > threshold


def batch_analyze(
    detector: RGPoisonDetector,
    prompts: List[str],
    max_tokens: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """
    Analyze multiple prompts in batch.
    
    Args:
        detector: RGPoisonDetector instance
        prompts: List of prompts to analyze
        max_tokens: Max tokens per generation
        verbose: Print progress
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"Processing {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        result = detector.analyze_generation(prompt, max_tokens=max_tokens)
        results.append(result)
        
        if verbose:
            print(f"  RTP: {result['rtp_detected']}, "
                  f"Score: {result['rtp_score']:.2f}, "
                  f"Gibberish: {detector.is_gibberish(result['text'])}")
    
    return results


# Example usage
if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Initializing detector...")
    detector = RGPoisonDetector(model, tokenizer, rtp_threshold=2.0)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "In Python, you can use",
        "The weather today is"
    ]
    
    print("\nAnalyzing prompts...")
    results = batch_analyze(detector, test_prompts, max_tokens=30)
    
    print("\nSummary:")
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {result['prompt']}")
        print(f"Generated: {result['text'][len(result['prompt']):100]}")
        print(f"RTP Detected: {result['rtp_detected']}")
        print(f"Mean Φ: {result['summary_stats']['phi']['mean']:.3f}")
        print(f"Mean κ: {result['summary_stats']['kappa']['mean']:.3f}")
