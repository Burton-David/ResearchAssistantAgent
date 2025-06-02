"""
Security utilities for handling sensitive data.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import getpass

logger = logging.getLogger(__name__)


class SecureConfig:
    """Handles secure storage and retrieval of sensitive configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize secure configuration handler."""
        self.config_path = config_path or Path.home() / ".research_assistant" / ".secure"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = None
        
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_file = self.config_path.parent / ".key"
        
        if key_file.exists():
            # Load existing key
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key from machine-specific data
            machine_id = self._get_machine_id()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=machine_id.encode(),
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
            
            # Save key with restricted permissions
            key_file.touch(mode=0o600)
            with open(key_file, "wb") as f:
                f.write(key)
                
            return key
            
    def _get_machine_id(self) -> str:
        """Get machine-specific identifier."""
        # Combine multiple sources for uniqueness
        sources = []
        
        # User home directory
        sources.append(str(Path.home()))
        
        # Username
        sources.append(os.getenv("USER", "default"))
        
        # Platform-specific ID
        try:
            if os.path.exists("/etc/machine-id"):
                with open("/etc/machine-id", "r") as f:
                    sources.append(f.read().strip())
        except:
            pass
            
        return "|".join(sources)
        
    def _get_cipher(self) -> Fernet:
        """Get or create cipher for encryption."""
        if not self._cipher:
            key = self._get_or_create_key()
            self._cipher = Fernet(key)
        return self._cipher
        
    def set_api_key(self, service: str, api_key: str) -> None:
        """Securely store an API key."""
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        # Load existing config
        config = self._load_config()
        
        # Encrypt API key
        cipher = self._get_cipher()
        encrypted_key = cipher.encrypt(api_key.encode())
        
        # Store encrypted key
        config[service] = {
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "timestamp": os.path.getmtime(__file__)
        }
        
        # Save config
        self._save_config(config)
        logger.info(f"Securely stored API key for {service}")
        
    def get_api_key(self, service: str) -> Optional[str]:
        """Retrieve an API key."""
        # First check environment variable
        env_key = f"{service.upper()}_API_KEY"
        if env_value := os.getenv(env_key):
            return env_value
            
        # Then check secure storage
        config = self._load_config()
        
        if service not in config:
            return None
            
        try:
            # Decrypt API key
            cipher = self._get_cipher()
            encrypted_key = base64.b64decode(config[service]["encrypted_key"])
            decrypted_key = cipher.decrypt(encrypted_key)
            return decrypted_key.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt API key for {service}: {e}")
            return None
            
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from disk."""
        if not self.config_path.exists():
            return {}
            
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
            
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to disk."""
        # Set restrictive permissions
        self.config_path.touch(mode=0o600)
        
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
            
    def configure_interactive(self) -> None:
        """Interactive configuration for API keys."""
        print("\nResearch Assistant Secure Configuration")
        print("=" * 40)
        
        services = [
            ("openai", "OpenAI API Key"),
            ("semantic_scholar", "Semantic Scholar API Key (optional)")
        ]
        
        for service, description in services:
            current = "configured" if self.get_api_key(service) else "not configured"
            print(f"\n{description}: {current}")
            
            if input("Configure? (y/n): ").lower() == "y":
                api_key = getpass.getpass(f"Enter {description}: ")
                if api_key:
                    self.set_api_key(service, api_key)
                    print(f"✓ {description} stored securely")
                    
        print("\nConfiguration complete!")
        

# Global secure config instance
_secure_config = None


def get_secure_config() -> SecureConfig:
    """Get global secure configuration instance."""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfig()
    return _secure_config