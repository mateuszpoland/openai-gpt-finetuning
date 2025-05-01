import re
from typing import Dict, Type, Optional, List
import importlib
import os
import glob
from .prompt_v4 import Prompt, PromptV4
from .prompt_v1 import PromptV1
from .prompt_v5 import PromptV5

class PromptStrategy:
    """Strategy for selecting and using prompt versions"""
    
    _prompt_classes: Dict[str, Type[Prompt]] = {}
    _loaded = False
    
    @classmethod
    def _load_prompt_classes(cls) -> None:
        """Load all available prompt classes"""
        if cls._loaded:
            return
            
        # Add known prompt classes
        cls._prompt_classes["v1"] = PromptV1
        cls._prompt_classes["v4"] = PromptV4
        cls._prompt_classes["v5"] = PromptV5
        
        # Dynamically load other prompt classes
        prompt_pattern = re.compile(r'prompt_v(\d+)\.py$')
        dir_path = os.path.dirname(os.path.abspath(__file__))
        
        for file_path in glob.glob(os.path.join(dir_path, "prompt_v*.py")):
            match = prompt_pattern.search(os.path.basename(file_path))
            if match:
                version = match.group(1)
                if f"v{version}" not in cls._prompt_classes:
                    try:
                        module_name = f".prompt_v{version}"
                        module = importlib.import_module(module_name, package="train.system_prompts")
                        class_name = f"PromptV{version}"
                        if hasattr(module, class_name):
                            prompt_class = getattr(module, class_name)
                            cls._prompt_classes[f"v{version}"] = prompt_class
                    except (ImportError, AttributeError) as e:
                        print(f"Failed to load prompt class for version {version}: {e}")
        
        cls._loaded = True
    
    @classmethod
    def get_prompt(cls, version: Optional[str] = None) -> Prompt:
        """
        Get a prompt by version or the latest version if none specified
        
        Args:
            version: The version string (e.g., 'v1', 'v2', etc.) or None for latest
            
        Returns:
            An instance of the requested prompt class
        """
        cls._load_prompt_classes()
        
        if version is None:
            # Find the latest version
            versions = sorted(cls._prompt_classes.keys(), key=lambda v: int(v.lstrip('v')))
            if not versions:
                raise ValueError("No prompt versions found")
            version = versions[-1]
        
        if version not in cls._prompt_classes:
            available = ", ".join(sorted(cls._prompt_classes.keys()))
            raise ValueError(f"Prompt version {version} not found. Available versions: {available}")
        
        return cls._prompt_classes[version]()
    
    @classmethod
    def list_versions(cls) -> List[str]:
        """List all available prompt versions"""
        cls._load_prompt_classes()
        return sorted(cls._prompt_classes.keys(), key=lambda v: int(v.lstrip('v')))