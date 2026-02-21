import os
import sys
import urllib.request
import urllib.error
import re
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def extract_file_id(drive_url: str) -> str:
    """Extract file ID from Google Drive sharing URL."""
    # Handle both formats: /file/d/FILE_ID/view and /open?id=FILE_ID
    if "/file/d/" in drive_url:
        start = drive_url.find("/file/d/") + len("/file/d/")
        end = drive_url.find("/view", start)
        if end == -1:
            end = drive_url.find("?", start)
        if end == -1:
            end = len(drive_url)
        return drive_url[start:end]
    elif "id=" in drive_url:
        start = drive_url.find("id=") + len("id=")
        end = drive_url.find("&", start)
        if end == -1:
            end = len(drive_url)
        return drive_url[start:end]
    else:
        raise ValueError("Could not extract file ID from Google Drive URL")


def get_direct_download_url(file_id: str) -> str:
    """Convert Google Drive sharing URL to direct download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(url: str, dest_path: str, progress: Progress) -> None:
    """Download file with progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    task_id = progress.add_task(f"Download {os.path.basename(dest_path)}", total=None)

    try:
        # Handle Google Drive virus scan warning
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        with urllib.request.urlopen(req) as response:
            # Check if this is a Google Drive virus scan page
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                html_content = response.read().decode('utf-8')
                if 'Virus scan warning' in html_content or 'Google Drive can\'t scan this file' in html_content:
                    print("Google Drive virus scan page detected. Extracting download link...")

                    # Extract the download form parameters using regex
                    confirm_match = re.search(r'<input[^>]*name="confirm"[^>]*value="([^"]*)"', html_content)
                    uuid_match = re.search(r'<input[^>]*name="uuid"[^>]*value="([^"]*)"', html_content)

                    if confirm_match and uuid_match:
                        confirm = confirm_match.group(1)
                        uuid_val = uuid_match.group(1)
                        # Construct the proper download URL
                        direct_url = f"https://drive.usercontent.google.com/download?id=1d3t_HihVboToBf8qHi2o4hqtaEn0mxuC&export=download&confirm={confirm}&uuid={uuid_val}"
                        print(f"Using direct download URL with extracted parameters")

                        # Retry with the direct URL
                        req2 = urllib.request.Request(direct_url, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        })
                        response = urllib.request.urlopen(req2)
                    else:
                        print("Could not extract download parameters from Google Drive page.")
                        print("Please download manually from:", url)
                        print("HTML content preview:")
                        print(html_content[:500] + "..." if len(html_content) > 500 else html_content)
                        return

            total_size = int(response.headers.get('Content-Length', 0))

            if total_size > 0:
                progress.update(task_id, total=total_size)

            with open(dest_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress.update(task_id, completed=min(downloaded, total_size))

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Google Drive download quota exceeded or file access denied.")
            print("Please download manually from:", url)
            return
        else:
            raise

    progress.update(task_id, completed=progress.tasks[task_id].total or progress.tasks[task_id].completed)


def main() -> int:
    """Download the model checkpoint from Google Drive."""
    drive_url = "https://drive.google.com/file/d/1d3t_HihVboToBf8qHi2o4hqtaEn0mxuC/view?usp=sharing"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(script_dir, "ultimate_tiled_multitask.pth")  # Save with proper name

    try:
        file_id = extract_file_id(drive_url)
        download_url = get_direct_download_url(file_id)

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            download_file(download_url, dest_path, progress)

        print(f"Checkpoint downloaded to: {dest_path}")
        return 0

    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())