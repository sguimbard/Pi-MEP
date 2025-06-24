import os
import subprocess
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import xarray as xr
import requests
from pathlib import Path
import hashlib
import time
import subprocess
import shutil
from typing import Optional, Union
import logging

"""
Download Utilities
A collection of functions for downloading and handling NetCDF files and figures in png format.

Usage:
    from netcdf_utils import download_netcdf_file, NetCDFDownloader
    
    # Simple usage
    file_path = download_netcdf_file(url, "data")
    
    # Advanced usage with custom settings
    downloader = NetCDFDownloader(data_dir="my_data", chunk_size=2*1024*1024)
    dataset = downloader.download_and_open(url)
    
    # Figure Download
    success = download_png_files_robust(url_pimep_web, region_widget.value, sat1_widget.value, insitu_widget.value)

"""


# Set up module logger
logger = logging.getLogger(__name__)

def download_files(base_url, output_dir="output/files/", 
                  file_extensions=None, skip_existing=True, force_redownload=False):
    """
    Download files from any URL with file existence checking - supports any file type
    
    Parameters:
    - base_url: Complete URL to scan for files
    - output_dir: Local output directory
    - file_extensions: List of extensions to download (e.g., ['.png', '.pdf', '.nc']) or None for all
    - skip_existing: If True, skip files that already exist locally
    - force_redownload: If True, redownload all files regardless of existence
    """
    
    # Ensure base_url ends with /
    if not base_url.endswith('/'):
        base_url += '/'
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process file extensions
    if file_extensions is None:
        file_extensions = []  # Empty list means all files
        extension_desc = "all file types"
    else:
        # Ensure extensions start with dot and are lowercase
        file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in file_extensions]
        extension_desc = f"files with extensions: {', '.join(file_extensions)}"
    
    print(f"🔍 Scanning directory: {base_url}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📄 Looking for: {extension_desc}")
    print(f"⚙️ Skip existing files: {skip_existing}")
    print(f"⚙️ Force redownload: {force_redownload}")
    
    try:
        # Get directory listing with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
        
        # Parse HTML to find files
        soup = BeautifulSoup(response.content, 'html.parser')
        all_files = []
        
        # Find files from links
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Skip parent directory links and other navigation
            if href in ['../', '../', '/', '#'] or href.startswith('?'):
                continue
            all_files.append(href)
        
        # Also look for files referenced in img, object, embed tags
        for tag in soup.find_all(['img', 'object', 'embed'], src=True):
            src = tag['src']
            if not src.startswith(('http', '//', '../')):
                all_files.append(src)
        
        # Remove duplicates and filter by extension
        all_files = list(set(all_files))
        target_files = []
        
        for file_path in all_files:
            # Get the file extension
            file_ext = Path(file_path).suffix.lower()
            
            # Filter by extension if specified
            if file_extensions:  # If extensions specified, filter
                if file_ext in file_extensions:
                    target_files.append(file_path)
            else:  # If no extensions specified, include files with any extension
                if file_ext:  # Only include files that have an extension
                    target_files.append(file_path)
        
        if not target_files:
            print(f"⚠️ No files found matching criteria")
            if all_files:
                print("Available files in directory:")
                for i, file in enumerate(all_files[:10], 1):  # Show first 10
                    print(f"  {i}. {file}")
                if len(all_files) > 10:
                    print(f"  ... and {len(all_files) - 10} more")
            return False
        
        print(f"📋 Found {len(target_files)} matching files on server")
        
        # Group files by extension for summary
        files_by_ext = {}
        for file_path in target_files:
            ext = Path(file_path).suffix.lower()
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(file_path)
        
        print(f"📊 File types found:")
        for ext, files in files_by_ext.items():
            print(f"  {ext}: {len(files)} files")
        
        # Check existing files and determine what to download
        files_to_download = []
        existing_files = []
        
        for file_path in target_files:
            # Get clean filename
            if file_path.startswith(('http://', 'https://')):
                filename = Path(urlparse(file_path).path).name
            else:
                filename = Path(file_path).name
            
            local_file_path = output_path / filename
            
            # Check if file exists
            if local_file_path.exists() and not force_redownload:
                if skip_existing:
                    existing_files.append(filename)
                    print(f"⏭️ Skipping existing file: {filename} ({local_file_path.stat().st_size:,} bytes)")
                    continue
                else:
                    # File exists but we're not skipping - add to download list
                    files_to_download.append((file_path, filename, local_file_path))
            else:
                # File doesn't exist or force_redownload is True
                files_to_download.append((file_path, filename, local_file_path))
        
        # Report what we found
        print(f"\n📊 File Status:")
        print(f"✅ Already exist locally: {len(existing_files)} files")
        print(f"📥 To be downloaded: {len(files_to_download)} files")
        
        if not files_to_download:
            print("🎉 All files already exist locally! No downloads needed.")
            return True
        
        # Show files to be downloaded
        print(f"\n📥 Files to download:")
        for i, (_, filename, _) in enumerate(files_to_download[:5], 1):
            print(f"  {i}. {filename}")
        if len(files_to_download) > 5:
            print(f"  ... and {len(files_to_download) - 5} more")
        
        # Download files with progress tracking
        downloaded_files = []
        failed_files = []
        
        for i, (file_path, filename, local_file_path) in enumerate(files_to_download, 1):
            try:
                # Construct full URL
                if file_path.startswith(('http://', 'https://')):
                    file_url = file_path
                else:
                    file_url = urljoin(base_url, file_path)
                
                print(f"📥 [{i}/{len(files_to_download)}] Downloading: {filename}")
                
                # Download with streaming for large files
                file_response = requests.get(file_url, timeout=60, stream=True, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                file_response.raise_for_status()
                
                # Save file
                with open(local_file_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                downloaded_files.append(filename)
                file_size = local_file_path.stat().st_size
                print(f"✅ Downloaded: {filename} ({file_size:,} bytes)")
                
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"❌ Failed to download {file_path}: {e}")
                continue
        
        # Final summary
        print(f"\n📊 Download Summary:")
        print(f"✅ Successfully downloaded: {len(downloaded_files)} files")
        print(f"⏭️ Already existed (skipped): {len(existing_files)} files")
        print(f"❌ Failed downloads: {len(failed_files)} files")
        
        # Count total files by extension in output directory
        all_local_files = list(output_path.glob('*'))
        local_by_ext = {}
        for f in all_local_files:
            if f.is_file():
                ext = f.suffix.lower()
                if ext not in local_by_ext:
                    local_by_ext[ext] = 0
                local_by_ext[ext] += 1
        
        print(f"📁 Total files in output directory by type:")
        for ext, count in sorted(local_by_ext.items()):
            print(f"  {ext}: {count} files")
        
        if downloaded_files:
            print(f"📁 Files saved to: {output_path.absolute()}")
            
        if failed_files:
            print("❌ Failed files:")
            for file, error in failed_files[:3]:  # Show first 3 failures
                print(f"  - {file}: {error}")
        
        return len(downloaded_files) > 0 or len(existing_files) > 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

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