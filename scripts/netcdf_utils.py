"""
NetCDF Download Utilities
A collection of functions for downloading and handling NetCDF files.

Usage:
    from netcdf_utils import download_netcdf_file, NetCDFDownloader
    
    # Simple usage
    file_path = download_netcdf_file(url, "data")
    
    # Advanced usage with custom settings
    downloader = NetCDFDownloader(data_dir="my_data", chunk_size=2*1024*1024)
    dataset = downloader.download_and_open(url)
"""

import xarray as xr
import requests
from pathlib import Path
import hashlib
import time
import subprocess
import shutil
from typing import Optional, Union
import logging

# Set up module logger
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """Set up logging for the module."""
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_wget_available() -> bool:
    """Check if wget is available on the system."""
    return shutil.which("wget") is not None

def download_with_wget(url: str, local_path: Path, timeout: int = 300) -> bool:
    """
    Download file using wget command.
    
    Args:
        url: URL to download from
        local_path: Local path to save file
        timeout: Timeout in seconds (default 5 minutes)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting wget download: {url}")
        start_time = time.time()
        
        # Build wget command
        cmd = [
            "wget",
            "--progress=bar:force",  # Force progress bar even when not in terminal
            "--show-progress",       # Show progress information
            "--timeout=30",          # Connection timeout
            f"--tries=3",           # Number of retries
            "--continue",           # Resume partial downloads
            "--output-document", str(local_path),  # Output file
            url
        ]
        
        # Run wget
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        download_time = time.time() - start_time
        
        if result.returncode == 0:
            file_size = local_path.stat().st_size
            download_speed = file_size / download_time / (1024*1024) if download_time > 0 else 0
            
            logger.info(f"wget download completed in {download_time:.1f}s")
            logger.info(f"Average speed: {download_speed:.1f} MB/s")
            logger.info(f"File size: {file_size / (1024*1024):.1f} MB")
            return True
        else:
            logger.error(f"wget failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            # Clean up partial file
            if local_path.exists():
                local_path.unlink()
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"wget download timed out after {timeout} seconds")
        if local_path.exists():
            local_path.unlink()
        return False
    except Exception as e:
        logger.error(f"wget download failed: {e}")
        if local_path.exists():
            local_path.unlink()
        return False

def download_with_progress(url: str, local_path: Path, chunk_size: int = 1024*1024) -> bool:
    """
    Download file with progress bar and improved error handling.
    
    Args:
        url: URL to download from
        local_path: Local path to save file
        chunk_size: Size of chunks to download (default 1MB)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import tqdm here to make it optional
        try:
            from tqdm import tqdm
            use_progress = True
        except ImportError:
            logger.warning("tqdm not available, downloading without progress bar")
            use_progress = False
        
        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file size for progress bar
        head_response = requests.head(url, timeout=30)
        head_response.raise_for_status()
        total_size = int(head_response.headers.get('content-length', 0))
        
        logger.info(f"Starting download: {url}")
        logger.info(f"File size: {total_size / (1024*1024):.1f} MB")
        
        start_time = time.time()
        
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            with open(local_path, 'wb') as file:
                if use_progress and total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=local_path.name
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Fallback without progress bar
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
        
        download_time = time.time() - start_time
        download_speed = total_size / download_time / (1024*1024) if download_time > 0 else 0
        
        logger.info(f"Download completed in {download_time:.1f}s")
        logger.info(f"Average speed: {download_speed:.1f} MB/s")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during download: {e}")
        if local_path.exists():
            local_path.unlink()
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        if local_path.exists():
            local_path.unlink()
        return False

def verify_file_integrity(file_path: Path, expected_size: Optional[int] = None) -> bool:
    """
    Verify downloaded file integrity.
    
    Args:
        file_path: Path to the downloaded file
        expected_size: Expected file size in bytes (optional)
    
    Returns:
        bool: True if file appears valid
    """
    if not file_path.exists():
        return False
    
    file_size = file_path.stat().st_size
    
    if expected_size and file_size != expected_size:
        logger.warning(f"File size mismatch: expected {expected_size}, got {file_size}")
        return False
    
    if file_size < 1024:  # Less than 1KB is suspicious for NetCDF
        logger.warning(f"File suspiciously small: {file_size} bytes")
        return False
    
    return True

def validate_netcdf(file_path: Path) -> bool:
    """
    Validate that a file is a proper NetCDF file by trying to open it.
    
    Args:
        file_path: Path to the NetCDF file
    
    Returns:
        bool: True if valid NetCDF file
    """
    try:
        with xr.open_dataset(file_path, decode_timedelta=False) as ds:
            # Just opening it is sufficient validation
            return True
    except Exception as e:
        logger.warning(f"NetCDF validation failed: {e}")
        return False

def download_netcdf_file(
    url: str, 
    data_dir: str = "data", 
    filename: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024*1024,
    use_wget: Optional[bool] = None,
    timeout: int = 300
) -> Optional[Path]:
    """
    Download NetCDF file to local directory with optimizations.
    
    Args:
        url: URL of the NetCDF file
        data_dir: Local directory to store the file
        filename: Optional custom filename (will extract from URL if not provided)
        force_download: Force re-download even if file exists
        chunk_size: Download chunk size in bytes (only used with requests)
        use_wget: Force use of wget (True), requests (False), or auto-detect (None)
        timeout: Timeout in seconds for wget downloads
    
    Returns:
        Path to downloaded file or None if failed
    """
    # Setup paths
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    if filename is None:
        filename = url.split('/')[-1]
    
    local_path = data_path / filename
    
    # Check if file already exists and is valid
    if local_path.exists() and not force_download:
        logger.info(f"File already exists: {local_path}")
        
        if validate_netcdf(local_path):
            logger.info("Existing file is valid NetCDF")
            return local_path
        else:
            logger.warning("Existing file appears corrupted, re-downloading...")
            local_path.unlink()
    
    # Determine download method
    if use_wget is None:
        use_wget = check_wget_available()
    
    if use_wget and check_wget_available():
        logger.info("Using wget for download")
        success = download_with_wget(url, local_path, timeout)
    else:
        if use_wget:
            logger.warning("wget requested but not available, falling back to requests")
        logger.info("Using requests for download")
        success = download_with_progress(url, local_path, chunk_size)
    
    if success and verify_file_integrity(local_path):
        logger.info(f"File successfully downloaded to: {local_path}")
        return local_path
    else:
        logger.error("Download failed or file integrity check failed")
        return None

class NetCDFDownloader:
    """
    A class for downloading and managing NetCDF files.
    
    Example:
        downloader = NetCDFDownloader(data_dir="my_data")
        dataset = downloader.download_and_open(url)
    """
    
    def __init__(self, data_dir: str = "data", chunk_size: int = 1024*1024, prefer_wget: bool = True):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Directory to store downloaded files
            chunk_size: Download chunk size in bytes (for requests method)
            prefer_wget: Whether to prefer wget over requests when available
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.prefer_wget = prefer_wget
        self.data_dir.mkdir(exist_ok=True)
        
    def download(self, url: str, filename: Optional[str] = None, force_download: bool = False, use_wget: Optional[bool] = None) -> Optional[Path]:
        """
        Download a NetCDF file.
        
        Args:
            url: URL of the NetCDF file
            filename: Optional custom filename
            force_download: Force re-download even if file exists
            use_wget: Override default wget preference
        """
        actual_use_wget = use_wget if use_wget is not None else self.prefer_wget
        
        return download_netcdf_file(
            url, 
            str(self.data_dir), 
            filename, 
            force_download, 
            self.chunk_size,
            actual_use_wget
        )
    
    def download_and_open(self, url: str, filename: Optional[str] = None, use_wget: Optional[bool] = None, **xr_kwargs) -> Optional[xr.Dataset]:
        """
        Download and open a NetCDF file in one step.
        
        Args:
            url: URL of the NetCDF file
            filename: Optional custom filename
            use_wget: Override default wget preference
            **xr_kwargs: Additional arguments to pass to xr.open_dataset
        
        Returns:
            xarray Dataset or None if failed
        """
        file_path = self.download(url, filename, use_wget=use_wget)
        
        if file_path:
            try:
                logger.info("Opening dataset with xarray...")
                # Set default decode_timedelta to False to avoid future warnings
                if 'decode_timedelta' not in xr_kwargs:
                    xr_kwargs['decode_timedelta'] = False
                return xr.open_dataset(file_path, **xr_kwargs)
            except Exception as e:
                logger.error(f"Failed to open NetCDF file: {e}")
                return None
        return None
    
    def list_files(self) -> list[Path]:
        """List all NetCDF files in the data directory."""
        return list(self.data_dir.glob("*.nc"))
    
    def clean_cache(self, keep_recent: int = 5):
        """
        Clean old files from cache, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent files to keep
        """
        files = sorted(self.list_files(), key=lambda p: p.stat().st_mtime, reverse=True)
        
        for file_path in files[keep_recent:]:
            logger.info(f"Removing old file: {file_path}")
            file_path.unlink()

# Convenience function for quick usage
def quick_download(url: str, data_dir: str = "data", decode_timedelta: bool = False, use_wget: Optional[bool] = None) -> Optional[xr.Dataset]:
    """
    Quick function to download and open a NetCDF file in one line.
    
    Args:
        url: URL of the NetCDF file
        data_dir: Directory to store the file
        decode_timedelta: Whether to decode timedelta variables (default False to avoid warnings)
        use_wget: Force use of wget (True), requests (False), or auto-detect (None)
    
    Returns:
        xarray Dataset or None if failed
    """
    downloader = NetCDFDownloader(data_dir)
    return downloader.download_and_open(url, use_wget=use_wget, decode_timedelta=decode_timedelta)

# Convenience function for speed comparison
def compare_download_methods(url: str, data_dir: str = "data") -> dict:
    """
    Compare download speeds between wget and requests.
    
    Args:
        url: URL of the NetCDF file
        data_dir: Directory to store test files
    
    Returns:
        dict: Results with timing information
    """
    results = {}
    
    # Test wget if available
    if check_wget_available():
        logger.info("Testing wget download speed...")
        start_time = time.time()
        wget_path = download_netcdf_file(url, data_dir, "test_wget.nc", force_download=True, use_wget=True)
        wget_time = time.time() - start_time
        
        if wget_path and wget_path.exists():
            file_size = wget_path.stat().st_size
            results['wget'] = {
                'time': wget_time,
                'speed_mbps': file_size / wget_time / (1024*1024),
                'file_size_mb': file_size / (1024*1024),
                'success': True
            }
            # Clean up test file
            wget_path.unlink()
        else:
            results['wget'] = {'success': False, 'error': 'Download failed'}
    else:
        results['wget'] = {'success': False, 'error': 'wget not available'}
    
    # Test requests
    logger.info("Testing requests download speed...")
    start_time = time.time()
    requests_path = download_netcdf_file(url, data_dir, "test_requests.nc", force_download=True, use_wget=False)
    requests_time = time.time() - start_time
    
    if requests_path and requests_path.exists():
        file_size = requests_path.stat().st_size
        results['requests'] = {
            'time': requests_time,
            'speed_mbps': file_size / requests_time / (1024*1024),
            'file_size_mb': file_size / (1024*1024),
            'success': True
        }
        # Clean up test file
        requests_path.unlink()
    else:
        results['requests'] = {'success': False, 'error': 'Download failed'}
    
    # Print comparison
    print("\n" + "="*50)
    print("DOWNLOAD SPEED COMPARISON")
    print("="*50)
    
    for method, result in results.items():
        if result['success']:
            print(f"{method.upper()}:")
            print(f"  Time: {result['time']:.1f}s")
            print(f"  Speed: {result['speed_mbps']:.1f} MB/s")
            print(f"  File size: {result['file_size_mb']:.1f} MB")
        else:
            print(f"{method.upper()}: {result['error']}")
    
    # Determine winner
    if results['wget']['success'] and results['requests']['success']:
        if results['wget']['speed_mbps'] > results['requests']['speed_mbps']:
            speedup = results['wget']['speed_mbps'] / results['requests']['speed_mbps']
            print(f"\n🏆 wget is {speedup:.1f}x faster than requests")
        else:
            speedup = results['requests']['speed_mbps'] / results['wget']['speed_mbps']
            print(f"\n🏆 requests is {speedup:.1f}x faster than wget")
    
    return results
