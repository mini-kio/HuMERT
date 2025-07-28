"""
HuMERT-300M Inference Script
"""

import torch
import torchaudio
import argparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from humert.model import HuMERTModel
from config.model_config import ModelConfig


class HuMERTInference:
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from {checkpoint_path}...")
        self.model, self.checkpoint = HuMERTModel.load_checkpoint(checkpoint_path, str(self.device))
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # Print model stats
        stats = self.model.get_model_stats()
        print(f"Model: {stats['total_parameters']/1e6:.1f}M parameters, {stats['model_size_mb']:.1f}MB")
    
    def load_audio(self, audio_path: str, target_length: float = None) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 24kHz if needed
        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)  # Remove channel dimension
        
        # Crop or pad to target length if specified
        if target_length is not None:
            target_samples = int(target_length * 24000)
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                padding = target_samples - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def extract_features(self, audio_path: str) -> torch.Tensor:
        """Extract encoded features from audio"""
        waveform = self.load_audio(audio_path)
        
        with torch.no_grad():
            features = self.model.extract_features(waveform)
        
        return features.cpu()
    
    def predict_speech(self, audio_path: str) -> dict:
        """Predict speech clusters"""
        waveform = self.load_audio(audio_path)
        
        with torch.no_grad():
            outputs = self.model.inference(waveform, task='speech')
        
        speech_logits = outputs['speech_output']
        predictions = torch.softmax(speech_logits, dim=-1)
        predicted_clusters = predictions.argmax(dim=-1)
        
        return {
            'predicted_clusters': predicted_clusters.cpu().numpy(),
            'probabilities': predictions.cpu().numpy(),
            'confidence': predictions.max(dim=-1)[0].cpu().numpy()
        }
    
    def predict_music(self, audio_path: str) -> dict:
        """Predict musical features"""
        waveform = self.load_audio(audio_path)
        
        with torch.no_grad():
            outputs = self.model.inference(waveform, task='music')
        
        music_features = outputs['music_output']
        
        return {
            'music_features': music_features.cpu().numpy(),
            'feature_dim': music_features.shape[-1]
        }
    
    def predict_dac(self, audio_path: str) -> dict:
        """Predict DAC codebook tokens"""
        waveform = self.load_audio(audio_path)
        
        with torch.no_grad():
            outputs = self.model.inference(waveform, task='dac')
        
        dac_logits = outputs['dac_output']  # [B, T, 9, 1024]
        
        # Get predicted tokens for each codebook
        predicted_tokens = dac_logits.argmax(dim=-1)  # [B, T, 9]
        probabilities = torch.softmax(dac_logits, dim=-1)
        confidence = probabilities.max(dim=-1)[0]  # [B, T, 9]
        
        return {
            'predicted_tokens': predicted_tokens.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'num_codebooks': dac_logits.shape[2],
            'vocab_size': dac_logits.shape[3]
        }
    
    def predict_all(self, audio_path: str) -> dict:
        """Run all tasks on the audio"""
        waveform = self.load_audio(audio_path)
        
        with torch.no_grad():
            outputs = self.model.inference(waveform, task='all')
        
        results = {
            'features': outputs['encoded_features'].cpu().numpy()
        }
        
        # Process each task output
        if 'dac_logits' in outputs:
            dac_logits = outputs['dac_logits']
            results['dac'] = {
                'predicted_tokens': dac_logits.argmax(dim=-1).cpu().numpy(),
                'confidence': torch.softmax(dac_logits, dim=-1).max(dim=-1)[0].cpu().numpy()
            }
        
        if 'speech_logits' in outputs:
            speech_logits = outputs['speech_logits']
            speech_probs = torch.softmax(speech_logits, dim=-1)
            results['speech'] = {
                'predicted_clusters': speech_probs.argmax(dim=-1).cpu().numpy(),
                'confidence': speech_probs.max(dim=-1)[0].cpu().numpy()
            }
        
        if 'music_features' in outputs:
            results['music'] = {
                'features': outputs['music_features'].cpu().numpy()
            }
        
        return results
    
    def batch_process(self, audio_paths: list, task: str = 'all', output_dir: str = None) -> dict:
        """Process multiple audio files"""
        results = {}
        
        for i, audio_path in enumerate(audio_paths):
            print(f"Processing {i+1}/{len(audio_paths)}: {audio_path}")
            
            try:
                if task == 'features':
                    result = {'features': self.extract_features(audio_path).numpy()}
                elif task == 'speech':
                    result = self.predict_speech(audio_path)
                elif task == 'music':
                    result = self.predict_music(audio_path)
                elif task == 'dac':
                    result = self.predict_dac(audio_path)
                else:  # 'all'
                    result = self.predict_all(audio_path)
                
                results[audio_path] = result
                
                # Save individual results if output directory specified
                if output_dir is not None:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    audio_name = Path(audio_path).stem
                    result_file = output_path / f"{audio_name}_{task}.json"
                    
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = self._convert_numpy_to_list(result)
                    with open(result_file, 'w') as f:
                        json.dump(json_result, f, indent=2)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results[audio_path] = {'error': str(e)}
        
        return results
    
    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description='HuMERT-300M Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--audio_list', type=str, help='Path to text file with list of audio files')
    parser.add_argument('--task', type=str, default='all', 
                       choices=['all', 'features', 'speech', 'music', 'dac'],
                       help='Task to perform')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = HuMERTInference(args.checkpoint, args.device)
    
    # Prepare audio file list
    if args.audio:
        audio_paths = [args.audio]
    elif args.audio_list:
        with open(args.audio_list, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either --audio or --audio_list must be provided")
    
    # Process audio files
    print(f"Processing {len(audio_paths)} audio file(s) with task: {args.task}")
    results = inference.batch_process(audio_paths, args.task, args.output)
    
    # Print summary
    successful = sum(1 for r in results.values() if 'error' not in r)
    print(f"\nProcessing complete: {successful}/{len(audio_paths)} files successful")
    
    # Save combined results
    if args.output:
        output_path = Path(args.output)
        combined_file = output_path / f"combined_results_{args.task}.json"
        json_results = inference._convert_numpy_to_list(results)
        with open(combined_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Combined results saved to {combined_file}")


if __name__ == '__main__':
    main()