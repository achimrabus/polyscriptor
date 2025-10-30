"""
Party HTR Engine with WSL Subprocess Isolation

Party is a PyTorch Lightning-based HTR framework with specific kraken dependencies
that conflict with the Windows kraken installation. This engine uses WSL subprocess
isolation to run party commands without affecting the main Windows environment.

Architecture:
- Windows Python: Main GUI application + existing kraken 6.0.2
- WSL venv_party_wsl: Isolated party installation with its own kraken version
- Communication: Subprocess calls with JSON for structured data exchange
"""

import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image


class PartyEngine:
    """
    Party HTR engine using WSL subprocess isolation.

    This engine runs party training and inference commands in an isolated WSL
    virtual environment to avoid dependency conflicts with the Windows kraken installation.
    """

    def __init__(self, model_path: Optional[str] = None, wsl_project_root: Optional[str] = None):
        """
        Initialize Party engine.

        Args:
            model_path: Path to party model (can be HuggingFace Hub ID or local path)
            wsl_project_root: WSL path to project root (default: /mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik)
        """
        self.model_path = model_path
        self.wsl_venv = "venv_party_wsl"

        # Default WSL project root
        if wsl_project_root is None:
            self.wsl_project_root = "/mnt/c/Users/Achim/Documents/TrOCR/dhlab-slavistik"
        else:
            self.wsl_project_root = wsl_project_root

    def _windows_to_wsl_path(self, windows_path: str) -> str:
        """
        Convert Windows path to WSL path.

        Args:
            windows_path: Windows path (e.g., C:\\Users\\...)

        Returns:
            WSL path (e.g., /mnt/c/Users/...)
        """
        # Convert backslashes to forward slashes
        path = windows_path.replace('\\', '/')

        # Convert drive letter (C: -> /mnt/c)
        if path[1:3] == ':/':
            drive = path[0].lower()
            path = f"/mnt/{drive}{path[2:]}"

        return path

    def _wsl_to_windows_path(self, wsl_path: str) -> str:
        """
        Convert WSL path to Windows path.

        Args:
            wsl_path: WSL path (e.g., /mnt/c/Users/...)

        Returns:
            Windows path (e.g., C:\\Users\\...)
        """
        # Convert /mnt/c to C:
        if wsl_path.startswith('/mnt/'):
            drive = wsl_path[5].upper()
            path = f"{drive}:{wsl_path[6:]}"
            return path.replace('/', '\\')

        return wsl_path

    def _run_wsl_command(self, command: str, check: bool = True, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """
        Run a command in WSL with party virtual environment activated.

        Args:
            command: Command to run (will be prefixed with venv activation)
            check: Whether to raise exception on non-zero exit code
            timeout: Command timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        # Build full command with venv activation
        full_command = f"cd {self.wsl_project_root} && source {self.wsl_venv}/bin/activate && {command}"

        # Execute via WSL
        result = subprocess.run(
            ["wsl", "bash", "-c", full_command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # Handle errors manually
        )

        if check and result.returncode != 0:
            raise RuntimeError(f"WSL command failed: {result.stderr}")

        return result.returncode, result.stdout, result.stderr

    def check_installation(self) -> Dict[str, any]:
        """
        Check if party is properly installed in WSL environment.

        Returns:
            Dict with installation status and version info
        """
        try:
            exit_code, stdout, stderr = self._run_wsl_command("party --version", check=False, timeout=10)

            if exit_code == 0:
                return {
                    "installed": True,
                    "version": stdout.strip(),
                    "error": None
                }
            else:
                return {
                    "installed": False,
                    "version": None,
                    "error": stderr.strip()
                }
        except Exception as e:
            return {
                "installed": False,
                "version": None,
                "error": str(e)
            }

    def train(self,
              train_list: str,
              val_list: str,
              output_dir: str,
              model_repo: Optional[str] = None,
              epochs: int = 50,
              batch_size: int = 8,
              workers: int = 4,
              additional_args: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Train a party model using WSL subprocess.

        Args:
            train_list: Path to training list file (.lst format)
            val_list: Path to validation list file (.lst format)
            output_dir: Directory to save model checkpoints
            model_repo: HuggingFace model repository to load from (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            workers: Number of dataloader workers
            additional_args: Additional command-line arguments

        Returns:
            Dict with training results
        """
        # Convert paths to WSL format
        train_list_wsl = self._windows_to_wsl_path(train_list)
        val_list_wsl = self._windows_to_wsl_path(val_list)
        output_dir_wsl = self._windows_to_wsl_path(output_dir)

        # Build party train command
        cmd_parts = [
            "party train",
            f"-t {train_list_wsl}",
            f"-e {val_list_wsl}",
            f"-o {output_dir_wsl}",
            f"--workers {workers}",
            f"--epochs {epochs}",
            f"--batch-size {batch_size}"
        ]

        # Add model repository if specified
        if model_repo:
            cmd_parts.append(f"--load-from-repo {model_repo}")

        # Add additional arguments
        if additional_args:
            cmd_parts.extend(additional_args)

        command = " ".join(cmd_parts)

        print(f"[PartyEngine] Starting training: {command}")

        try:
            exit_code, stdout, stderr = self._run_wsl_command(command, check=True, timeout=None)

            return {
                "success": True,
                "output": stdout,
                "error": stderr,
                "exit_code": exit_code
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "exit_code": -1
            }

    def transcribe(self, image_path: str, model_path: Optional[str] = None) -> Tuple[str, float]:
        """
        Transcribe a single line image using party inference.

        Args:
            image_path: Path to line image
            model_path: Path to party model (uses self.model_path if not specified)

        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            raise ValueError("No model path specified. Set model_path in constructor or pass as argument.")

        # Convert paths to WSL format
        image_path_wsl = self._windows_to_wsl_path(image_path)
        model_path_wsl = self._windows_to_wsl_path(model_path) if Path(model_path).exists() else model_path

        # Create temporary file for JSON output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            tmp_path_wsl = self._windows_to_wsl_path(tmp_path)

        try:
            # Build party inference command with JSON output
            command = f"party transcribe -m {model_path_wsl} {image_path_wsl} --format json -o {tmp_path_wsl}"

            exit_code, stdout, stderr = self._run_wsl_command(command, check=False, timeout=60)

            if exit_code != 0:
                raise RuntimeError(f"Party transcribe failed: {stderr}")

            # Read JSON output
            with open(tmp_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # Extract text and confidence
            text = result.get('text', '')
            confidence = result.get('confidence', 1.0)

            return text, confidence

        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except:
                pass

    def transcribe_batch(self, image_paths: List[str], model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Transcribe multiple line images in batch.

        Args:
            image_paths: List of paths to line images
            model_path: Path to party model (uses self.model_path if not specified)

        Returns:
            List of tuples (transcribed_text, confidence_score)
        """
        if model_path is None:
            model_path = self.model_path

        if model_path is None:
            raise ValueError("No model path specified. Set model_path in constructor or pass as argument.")

        # Convert paths to WSL format
        image_paths_wsl = [self._windows_to_wsl_path(p) for p in image_paths]
        model_path_wsl = self._windows_to_wsl_path(model_path) if Path(model_path).exists() else model_path

        # Create temporary files for batch processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_list:
            tmp_list_path = tmp_list.name
            tmp_list_path_wsl = self._windows_to_wsl_path(tmp_list_path)

            # Write image paths to list file
            for img_path_wsl in image_paths_wsl:
                tmp_list.write(f"{img_path_wsl}\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
            tmp_out_path_wsl = self._windows_to_wsl_path(tmp_out_path)

        try:
            # Build party batch inference command
            command = f"party transcribe -m {model_path_wsl} --list {tmp_list_path_wsl} --format json -o {tmp_out_path_wsl}"

            exit_code, stdout, stderr = self._run_wsl_command(command, check=False, timeout=600)

            if exit_code != 0:
                raise RuntimeError(f"Party batch transcribe failed: {stderr}")

            # Read JSON output
            with open(tmp_out_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            # Extract text and confidence for each result
            transcriptions = []
            for result in results:
                text = result.get('text', '')
                confidence = result.get('confidence', 1.0)
                transcriptions.append((text, confidence))

            return transcriptions

        finally:
            # Clean up temporary files
            try:
                Path(tmp_list_path).unlink()
                Path(tmp_out_path).unlink()
            except:
                pass


# Example usage and testing
if __name__ == "__main__":
    # Initialize party engine
    engine = PartyEngine()

    # Check installation
    status = engine.check_installation()
    print(f"Party installation status: {status}")

    if status["installed"]:
        print(f"Party version: {status['version']}")
    else:
        print(f"Party not installed or error: {status['error']}")
