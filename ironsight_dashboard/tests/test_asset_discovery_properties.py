"""
Property-based tests for asset discovery completeness.
Tests Property 15: Asset Discovery Completeness
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from asset_discovery import AssetDiscoveryManager, AssetInventory


# Feature: ironsight-command-center, Property 15: Asset Discovery Completeness
@settings(max_examples=100)
@given(
    config_files=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=0, max_size=5
    ),
    model_files=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=0, max_size=3
    ),
    script_files=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=0, max_size=4
    )
)
def test_asset_discovery_completeness(config_files, model_files, script_files):
    """
    Feature: ironsight-command-center, Property 15: Asset Discovery Completeness
    For any directory structure with config files, model files, and scripts,
    the asset discovery SHALL find all files of the expected types.
    **Validates: Requirements 15.1**
    """
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create subdirectories
        config_dir = temp_path / 'config'
        models_dir = temp_path / 'models'
        scripts_dir = temp_path / 'scripts'
        src_dir = temp_path / 'src'
        
        config_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)
        src_dir.mkdir(exist_ok=True)
        
        # Create config files with various extensions
        created_config_files = []
        config_extensions = ['.yaml', '.yml', '.json', '.toml']
        for i, filename in enumerate(config_files):
            if filename.strip():  # Skip empty filenames
                ext = config_extensions[i % len(config_extensions)]
                config_file = config_dir / f"{filename}{ext}"
                config_file.write_text("# Test config file")
                created_config_files.append(config_file)
        
        # Create model files with various extensions
        created_model_files = []
        model_extensions = ['.pth', '.pt', '.onnx', '.bin']
        for i, filename in enumerate(model_files):
            if filename.strip():  # Skip empty filenames
                ext = model_extensions[i % len(model_extensions)]
                model_file = models_dir / f"{filename}{ext}"
                model_file.write_text("# Test model file")
                created_model_files.append(model_file)
        
        # Create script files (ensure unique names to avoid duplicates)
        created_script_files = []
        unique_script_names = list(set(script_files))  # Remove duplicates from input
        for i, filename in enumerate(unique_script_names):
            if filename.strip():  # Skip empty filenames
                # Create unique scripts in both scripts/ and src/ directories
                script_file1 = scripts_dir / f"{filename}_scripts.py"
                script_file2 = src_dir / f"{filename}_src.py"
                
                script_file1.write_text("# Test script file")
                script_file2.write_text("# Test script file")
                
                created_script_files.extend([script_file1, script_file2])
        
        # Run asset discovery
        manager = AssetDiscoveryManager(temp_path)
        inventory = manager.scan_assets()
        
        # Verify all created config files were discovered
        discovered_config_paths = {str(p) for p in inventory.config_files}
        for config_file in created_config_files:
            assert str(config_file) in discovered_config_paths, f"Config file not discovered: {config_file}"
        
        # Verify all created model files were discovered
        discovered_model_paths = {str(p) for p in inventory.model_files}
        for model_file in created_model_files:
            assert str(model_file) in discovered_model_paths, f"Model file not discovered: {model_file}"
        
        # Verify all created script files were discovered
        discovered_script_paths = {str(p) for p in inventory.utility_scripts}
        for script_file in created_script_files:
            assert str(script_file) in discovered_script_paths, f"Script file not discovered: {script_file}"
        
        # Verify counts match expectations
        assert len(inventory.config_files) == len(created_config_files)
        assert len(inventory.model_files) == len(created_model_files)
        assert len(inventory.utility_scripts) == len(created_script_files)


@settings(max_examples=50)
@given(
    directory_depth=st.integers(min_value=1, max_value=3),
    files_per_level=st.integers(min_value=0, max_value=3)
)
def test_asset_discovery_recursive_search(directory_depth, files_per_level):
    """
    Feature: ironsight-command-center, Property 15: Asset Discovery Completeness
    For any nested directory structure, asset discovery SHALL find files
    at all levels of nesting using recursive search.
    **Validates: Requirements 15.1**
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create nested directory structure
        created_files = []
        current_dir = temp_path
        
        for level in range(directory_depth):
            # Create subdirectory
            current_dir = current_dir / f"level_{level}"
            current_dir.mkdir(exist_ok=True)
            
            # Create files at this level
            for i in range(files_per_level):
                config_file = current_dir / f"config_{level}_{i}.yaml"
                config_file.write_text("# Test config")
                created_files.append(config_file)
        
        # Run asset discovery
        manager = AssetDiscoveryManager(temp_path)
        inventory = manager.scan_assets()
        
        # Verify all files were discovered regardless of nesting level
        discovered_paths = {str(p) for p in inventory.config_files}
        for created_file in created_files:
            assert str(created_file) in discovered_paths, f"Nested file not discovered: {created_file}"
        
        # Verify count matches
        assert len(inventory.config_files) == len(created_files)


@settings(max_examples=50)
@given(
    hidden_files=st.integers(min_value=0, max_value=3),
    normal_files=st.integers(min_value=1, max_value=3)
)
def test_asset_discovery_ignores_hidden_files(hidden_files, normal_files):
    """
    Feature: ironsight-command-center, Property 15: Asset Discovery Completeness
    For any directory containing both hidden and normal files,
    asset discovery SHALL ignore hidden files and directories.
    **Validates: Requirements 15.1**
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create hidden directory
        hidden_dir = temp_path / '.hidden'
        hidden_dir.mkdir(exist_ok=True)
        
        # Create normal directory
        normal_dir = temp_path / 'normal'
        normal_dir.mkdir(exist_ok=True)
        
        # Create hidden files (should be ignored)
        for i in range(hidden_files):
            hidden_file = hidden_dir / f"hidden_config_{i}.yaml"
            hidden_file.write_text("# Hidden config")
        
        # Create normal files (should be discovered)
        created_normal_files = []
        for i in range(normal_files):
            normal_file = normal_dir / f"normal_config_{i}.yaml"
            normal_file.write_text("# Normal config")
            created_normal_files.append(normal_file)
        
        # Run asset discovery
        manager = AssetDiscoveryManager(temp_path)
        inventory = manager.scan_assets()
        
        # Verify only normal files were discovered
        discovered_paths = {str(p) for p in inventory.config_files}
        
        # Check that no hidden files were discovered
        for discovered_path in discovered_paths:
            assert '.hidden' not in discovered_path, f"Hidden file was discovered: {discovered_path}"
        
        # Check that all normal files were discovered
        for normal_file in created_normal_files:
            assert str(normal_file) in discovered_paths, f"Normal file not discovered: {normal_file}"
        
        # Verify count matches only normal files
        assert len(inventory.config_files) == len(created_normal_files)


def test_asset_discovery_empty_directory():
    """
    Feature: ironsight-command-center, Property 15: Asset Discovery Completeness
    For any empty directory, asset discovery SHALL return empty inventory
    without errors.
    **Validates: Requirements 15.1**
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run asset discovery on empty directory
        manager = AssetDiscoveryManager(temp_path)
        inventory = manager.scan_assets()
        
        # Verify all lists are empty
        assert len(inventory.config_files) == 0
        assert len(inventory.font_files) == 0
        assert len(inventory.utility_scripts) == 0
        assert len(inventory.model_files) == 0
        assert len(inventory.data_directories) == 0


def test_asset_inventory_serialization():
    """
    Feature: ironsight-command-center, Property 15: Asset Discovery Completeness
    For any AssetInventory, serialization to dictionary SHALL preserve
    all file paths correctly.
    **Validates: Requirements 15.1**
    """
    # Create test inventory with normalized paths
    inventory = AssetInventory()
    inventory.config_files = [Path('/test/config.yaml'), Path('/test/config.json')]
    inventory.model_files = [Path('/test/model.pth')]
    inventory.utility_scripts = [Path('/test/script.py')]
    
    # Serialize to dictionary
    inventory_dict = inventory.to_dict()
    
    # Verify structure
    assert 'config_files' in inventory_dict
    assert 'model_files' in inventory_dict
    assert 'utility_scripts' in inventory_dict
    assert 'font_files' in inventory_dict
    assert 'data_directories' in inventory_dict
    
    # Verify content (normalize paths for cross-platform compatibility)
    expected_config_paths = ['/test/config.yaml', '/test/config.json']
    actual_config_paths = [str(Path(p).as_posix()) for p in inventory_dict['config_files']]
    assert actual_config_paths == expected_config_paths
    
    expected_model_paths = ['/test/model.pth']
    actual_model_paths = [str(Path(p).as_posix()) for p in inventory_dict['model_files']]
    assert actual_model_paths == expected_model_paths
    
    expected_script_paths = ['/test/script.py']
    actual_script_paths = [str(Path(p).as_posix()) for p in inventory_dict['utility_scripts']]
    assert actual_script_paths == expected_script_paths
    
    assert inventory_dict['font_files'] == []
    assert inventory_dict['data_directories'] == []


if __name__ == "__main__":
    # Run a simple test
    test_asset_discovery_empty_directory()
    print("Basic asset discovery test passed!")