import numpy as np
import selfies as sf
import torch
import rdkit
import re
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.DataStructs import TanimotoSimilarity


class SelfiesMetrics:
    def __init__(self, idx_to_token):
        self.idx_to_token = idx_to_token
        self._get_special_tokens()

    def _get_special_tokens(self):
        # Create reverse mapping from token to idx
        self.token_to_idx = {v: k for k, v in self.idx_to_token.items()}

        # Your specific special tokens
        PAD_TOKEN = "<PAD>"
        UNK_TOKEN = "<UNK>"
        START_TOKEN = "<START>"
        END_TOKEN = "<END>"

        # Get token IDs
        self.PAD = self.token_to_idx.get(PAD_TOKEN)
        self.UNK = self.token_to_idx.get(UNK_TOKEN)
        self.START = self.token_to_idx.get(START_TOKEN)
        self.EOS = self.token_to_idx.get(END_TOKEN)

        # Verify all tokens exist in vocabulary
        assert self.PAD is not None, f"PAD token '{PAD_TOKEN}' not found in vocabulary"
        assert self.UNK is not None, f"UNK token '{UNK_TOKEN}' not found in vocabulary"
        assert (
            self.START is not None
        ), f"START token '{START_TOKEN}' not found in vocabulary"
        assert self.EOS is not None, f"EOS token '{END_TOKEN}' not found in vocabulary"

    def standardize_inputs(self, pred, gt):
        """
        Standardize inputs for efficient metric calculation.

        Args:
            pred: torch.Tensor [batch_size, seq_len] - predicted token IDs (long)
            gt: List[str] - ground truth SELFIES strings
        """
        # Convert predictions to numpy for efficient processing
        self.pred_tokens = pred.detach().cpu().numpy().astype(int)  # [B, seq_len] int

        # Validation for empty predictions
        if self.pred_tokens.size == 0:
            raise ValueError("Predictions tensor is empty")

        # Decode predictions to SELFIES strings
        self.pred_to_selfies()

        # Convert predicted SELFIES to RDKit Mol objects
        self.pred_mol = self.selfies_to_mol(self.pred_selfies)

        # GT is already strings
        self.gt_selfies = gt

        # Convert GT SELFIES to RDKit Mol objects
        self.gt_mol = self.selfies_to_mol(self.gt_selfies)

        # Tokenize GT strings to match pred format
        self.selfies_to_token()

        self.gt_tokens = np.array(self.gt_tokens, dtype=np.int64)  # [B, seq_len] int64

    def selfies_to_token(self):
        """Properly tokenize SELFIES strings using selfies library"""
        self.gt_tokens = []
        self.gt_selfies_tokens = []

        for selfie in self.gt_selfies:
            # Use selfies library to split properly
            try:
                selfies_tokens = list(sf.split_selfies(selfie))
            except Exception:
                # Fallback if SELFIES parsing fails
                selfies_tokens = []

            # Store the string tokens
            self.gt_selfies_tokens.append(selfies_tokens)

            tokens = []
            for token in selfies_tokens:
                token_id = self.token_to_idx.get(token, self.UNK)
                tokens.append(token_id)

            # Pad to match pred length
            pred_len = self.pred_tokens.shape[1]
            if len(tokens) < pred_len:
                tokens.extend([self.PAD] * (pred_len - len(tokens)))
            else:
                tokens = tokens[:pred_len]
            self.gt_tokens.append(tokens)

    def pred_to_selfies(self):
        self.pred_selfies = []
        self.pred_selfies_tokens = []
        for seq in self.pred_tokens:
            tokens = []
            for token_id in seq:

                if token_id == self.EOS:
                    break
                if token_id not in [
                    self.PAD,
                    self.START,
                ]:  # Skip padding AND start token
                    token_str = self.idx_to_token.get(token_id, "<UNK>")
                    tokens.append(token_str)
            self.pred_selfies_tokens.append(tokens)
            self.pred_selfies.append("".join(tokens))

    def calculate_sequence_accuracy(self):
        """
        Calculate sequence-level accuracy between predicted and ground truth tokens,
        using pre-computed tokenized sequences for efficiency.

        Returns
        -------
        tuple
            (np.ndarray, float): Array of sequence accuracies and average accuracy
            - Array shape: [batch_size]
            - Average: single float value
        """
        accuracies = []
        # counter = 0 

        for pred_tokens, gt_tokens in zip(
            self.pred_selfies_tokens, self.gt_selfies_tokens
        ):
            # if counter < 4:
            #     print("Predicted SELFIES:", pred_tokens)
            #     print("Ground Truth SELFIES:", gt_tokens)
            #     self.pred_tokens, self.gt_tokens
            #     counter += 1
            # Compare token sequences directly
            accuracy = float(pred_tokens == gt_tokens)
            accuracies.append(accuracy)

        accuracies_array = np.array(accuracies)
        return accuracies_array, accuracies_array.mean()

    def calculate_token_accuracy(self):
        """
        Calculate token-level accuracy between predicted and ground truth tokens,
        using pre-computed tokenized sequences for efficiency.

        Returns
        -------
        tuple
            (np.ndarray, float): Array of token accuracies and average accuracy
            - Array shape: [batch_size]
            - Average: single float value
        """
        accuracies = []

        for pred_tokens, gt_tokens in zip(
            self.pred_selfies_tokens, self.gt_selfies_tokens
        ):
            n = min(len(pred_tokens), len(gt_tokens))
            if n == 0:
                accuracies.append(0.0)
                continue

            matches = sum(1 for i in range(n) if pred_tokens[i] == gt_tokens[i])
            accuracy = matches / n
            accuracies.append(accuracy)

        accuracies_array = np.array(accuracies)
        return accuracies_array, accuracies_array.mean()

    def selfies_to_mol(self, selfies_list):
        """
        Convert list of SELFIES strings to RDKit Mol objects.

        Parameters
        ----------
        selfies_list : List[str]
            List of SELFIES strings to convert

        Returns
        -------
        List[Optional[Chem.Mol]]
            List of RDKit Mol objects or None if conversion fails
        """
        mol_list = []
        for selfies in selfies_list:
            try:
                smiles = sf.decoder(selfies)
                mol = Chem.MolFromSmiles(smiles)
                mol_list.append(mol)
            except Exception:
                mol_list.append(None)
        return mol_list

    def calculate_tanimoto_similarity(self):
        """
        Calculate Tanimoto similarity between predicted and ground truth molecules,
        using pre-computed RDKit Mol objects for efficiency.

        Returns
        -------
        tuple
            (np.ndarray, float): Array of Tanimoto similarities and average similarity
            - Array shape: [batch_size]
            - Average: single float value
        """
        similarities = []

        for pred_mol, gt_mol in zip(self.pred_mol, self.gt_mol):
            if pred_mol is None or gt_mol is None:
                similarities.append(0.0)
                continue

            try:
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fp1 = gen.GetCountFingerprint(pred_mol)
                fp2 = gen.GetCountFingerprint(gt_mol)
                similarity = TanimotoSimilarity(fp1, fp2)
                similarities.append(similarity)
            except Exception:
                similarities.append(0.0)

        similarities_array = np.array(similarities)
        return similarities_array, similarities_array.mean()

    def calculate_valid_mol(self):
        """
        Calculate molecular validity for predicted molecules.
        Returns 1.0 if molecule is valid (not None), 0.0 if invalid (None).

        Returns
        -------
        tuple
            (np.ndarray, float): Array of validity scores and average validity
            - Array shape: [batch_size]
            - Average: single float value (fraction of valid molecules)
        """
        validities = []

        for pred_mol in self.pred_mol:
            validity = 1.0 if pred_mol is not None else 0.0
            validities.append(validity)

        validities_array = np.array(validities)
        return validities_array, validities_array.mean()
        
    def calculate_hdi(self, mol):
        """
        Calculate Hydrogen Deficiency Index (HDI) for a molecule.

        Parameters
        ----------
        mol : RDKit Mol object or None
            Molecule to calculate HDI for

        Returns
        -------
        float
            HDI value, or np.nan if calculation fails
        """
        if mol is None:
            return np.nan

        try:
            # Add explicit hydrogens to the molecule for accurate counting
            mol_with_h = Chem.AddHs(mol)
            
            num_c = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetAtomicNum() == 6)
            num_h = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetAtomicNum() == 1)  # Count explicit H atoms
            num_n = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetAtomicNum() == 7)
            num_x = sum(
                1 for atom in mol_with_h.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]
            )

            # HDI formula: rings + double bond equivalents
            hdi = (2 * num_c + 2 + num_n - num_h - num_x) / 2
            
            return float(hdi)
        except Exception:
            return np.nan

    def calculate_delta_hdi(self):
        """
        Calculate delta HDI (HDI_pred - HDI_gt) between predicted and ground truth molecules.

        Returns
        -------
        tuple
            (np.ndarray, float): Array of delta HDI values and average delta HDI
            - Array shape: [batch_size]
            - Average: single float value
        """
        delta_hdis = []

        for pred_mol, gt_mol in zip(self.pred_mol, self.gt_mol):
            hdi_pred = self.calculate_hdi(pred_mol)
            hdi_gt = self.calculate_hdi(gt_mol)

            if np.isnan(hdi_pred) or np.isnan(hdi_gt):
                delta_hdi = np.nan
            else:
                delta_hdi = hdi_pred - hdi_gt

            delta_hdis.append(delta_hdi)

        delta_hdis_array = np.array(delta_hdis)
        # Use nanmean to handle NaN values when computing average
        return delta_hdis_array, np.nanmean(delta_hdis_array)

    def calculate_size_accuracy(self):
        """
        Calculate size accuracy between predicted and ground truth molecules.
        Returns normalized size accuracy (1 - normalized_size_difference).
        
        Returns
        -------
        tuple
            (np.ndarray, float): Array of size accuracies and average size accuracy
            - Array shape: [batch_size]
            - Average: single float value
        """
        pred_sizes = []
        gt_sizes = []
        
        # Calculate molecule sizes (atom counts)
        for pred_mol in self.pred_mol:
            if pred_mol is not None:
                pred_sizes.append(len(pred_mol.GetAtoms()))
            else:
                pred_sizes.append(0)
                
        for gt_mol in self.gt_mol:
            if gt_mol is not None:
                gt_sizes.append(len(gt_mol.GetAtoms()))
            else:
                gt_sizes.append(0)
        
        # Calculate size differences
        size_differences = [abs(pred - gt) for pred, gt in zip(pred_sizes, gt_sizes)]
        
        # Normalize by maximum size to prevent division by zero
        max_size = max(gt_sizes) if gt_sizes and max(gt_sizes) > 0 else 1
        normalized_differences = [diff / max_size for diff in size_differences]
        
        # Convert to accuracy (1 - normalized_difference)
        size_accuracies = [1 - norm_diff for norm_diff in normalized_differences]
        
        size_accuracies_array = np.array(size_accuracies)
        return size_accuracies_array, size_accuracies_array.mean()

    def compute_metrics(self, pred, gt, phase='None'):
        self.standardize_inputs(pred, gt)
        # Vars available after standardization
        # self.pred_tokens, self.gt_tokens, self.pred_selfies, self.gt_selfies
        # self.pred_selfies_tokens, self.gt_selfies_tokens (list of string tokens)
        # self.pred_mol, self.gt_mol (list of RDKit Mol objects)
        # dtype = np.int64, # shape = [B, seq_len]


        self.phase = phase

        if self.phase == "val":
        # Calculate all metrics
            seq_acc_array, seq_acc_mean = self.calculate_sequence_accuracy()
            token_acc_array, token_acc_mean = self.calculate_token_accuracy()
        # tanimoto_array, tanimoto_mean = self.calculate_tanimoto_similarity()
        # valid_mol_array, valid_mol_mean = self.calculate_valid_mol()
        # delta_hdi_array, delta_hdi_mean = self.calculate_delta_hdi()
        # size_acc_array, size_acc_mean = self.calculate_size_accuracy()

        # Return results dictionary
        results = {
            "sequence_accuracy": {"array": seq_acc_array, "mean": seq_acc_mean},
            "token_accuracy": {"array": token_acc_array, "mean": token_acc_mean},
            # "tanimoto_similarity": {"array": tanimoto_array, "mean": tanimoto_mean},
            # "valid_mol": {"array": valid_mol_array, "mean": valid_mol_mean},
            # "delta_hdi": {"array": delta_hdi_array, "mean": delta_hdi_mean},
            # "size_accuracy": {"array": size_acc_array, "mean": size_acc_mean},
        }

        return results


def test_selfies_metrics():
    """Test function to verify SelfiesMetrics functionality"""
    print("Testing SelfiesMetrics...")

    # Create a simple vocabulary for testing
    vocab = {
        0: "<PAD>",
        1: "<START>",
        2: "<END>",
        3: "<UNK>",
        4: "[C]",
        5: "[=C]",
        6: "[O]",
        7: "[=O]",
        8: "[N]",
        9: "[Ring1]",
        10: "[Branch1]",
    }

    # Initialize metrics
    metrics = SelfiesMetrics(vocab)

    # Create sample predictions (batch_size=3, seq_len=5)
    # Pred 1: [START, C, =O, END, PAD] -> [C][=O]
    # Pred 2: [START, C, C, O, END] -> [C][C][O]
    # Pred 3: [START, N, Ring1, END, PAD] -> [N][Ring1]
    pred_tokens = torch.tensor(
        [
            [1, 4, 7, 2, 0, 0],  # [START, C, =O, END, PAD]
            [1, 5, 5, 5, 5, 2],  # [START, C, C, O, END]
            [1, 8, 9, 2, 0, 0],  # [START, N, Ring1, END, PAD]
        ],
        dtype=torch.long,
    )

    # Ground truth SELFIES strings
    gt_selfies = [
        "[C][=O]",  # Should match pred 1
        "[C][C][N]",  # Different from pred 2
        "[N][Ring1]",  # Should match pred 3
    ]

    print(f"Predictions shape: {pred_tokens.shape}")
    print(f"Ground truth: {gt_selfies}")

    # Compute metrics
    try:
        results = metrics.compute_metrics(pred_tokens, gt_selfies)

        print("\n=== Results ===")
        print(f"Sequence Accuracy: {results['sequence_accuracy']['mean']:.3f}")
        print(f"Token Accuracy: {results['token_accuracy']['mean']:.3f}")
        print(f"Tanimoto Similarity: {results['tanimoto_similarity']['mean']:.3f}")
        print(f"Valid Molecules: {results['valid_mol']['mean']:.3f}")
        print(f"Delta HDI: {results['delta_hdi']['mean']:.3f}")
        print(f"Size Accuracy: {results['size_accuracy']['mean']:.3f}")

        print(f"\nPer-sample results:")
        for i in range(len(gt_selfies)):
            print(f"Sample {i+1}:")
            print(f"  Seq Acc: {results['sequence_accuracy']['array'][i]:.3f}")
            print(f"  Token Acc: {results['token_accuracy']['array'][i]:.3f}")
            print(f"  Tanimoto: {results['tanimoto_similarity']['array'][i]:.3f}")
            print(f"  Valid Mol: {results['valid_mol']['array'][i]:.3f}")
            print(f"  Delta HDI: {results['delta_hdi']['array'][i]:.3f}")
            print(f"  Size Acc: {results['size_accuracy']['array'][i]:.3f}")

        print(f"\nGenerated SELFIES:")
        for i, selfie in enumerate(metrics.pred_selfies):
            print(f"  Pred {i+1}: '{selfie}' vs GT: '{gt_selfies[i]}'")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_selfies_metrics()
