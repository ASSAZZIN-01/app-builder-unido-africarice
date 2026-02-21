import os
import urllib.request
import zipfile

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


ZIP_FILES = [
    "images-20251223T085919Z-3-001.zip",
    "images-20251223T085919Z-3-002.zip",
    "images-20251223T085919Z-3-003.zip",
    "images-20251223T085919Z-3-004.zip",
    "images-20251223T085919Z-3-005.zip",
    "images-20251223T085919Z-3-006.zip",
    "images-20251223T085919Z-3-007.zip",
    "images-20251223T085919Z-3-008.zip",
]


def download_file(url: str, dest_path: str, progress: Progress) -> None:
    """Download a single zip file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"Already exists, skipping: {dest_path}")
        return
    task_id = progress.add_task(f"Download {os.path.basename(dest_path)}", total=None)

    def reporthook(block_num, block_size, total_size):
        """Update progress as urllib downloads the file."""
        if total_size and total_size > 0:
            progress.update(task_id, total=total_size)
        downloaded = block_num * block_size
        progress.update(task_id, completed=min(downloaded, total_size or downloaded))

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    progress.update(task_id, completed=progress.tasks[task_id].total or progress.tasks[task_id].completed)


def unzip_file(zip_path: str, dest_dir: str, progress: Progress) -> None:
    """Extract a zip file into the destination directory."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        task_id = progress.add_task(f"Extract {os.path.basename(zip_path)}", total=len(members))
        for member in members:
            zf.extract(member, dest_dir)
            progress.update(task_id, advance=1)


def main() -> int:
    """Download and extract all image archives into Data/images/."""
    # Use current working directory and create/find Data folder there
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "Data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    base_url = "https://storage.googleapis.com/unido-afririce/"
    with progress:
        for zip_name in ZIP_FILES:
            url = f"{base_url}{zip_name}"
            zip_path = os.path.join(data_dir, zip_name)
            download_file(url, zip_path, progress)
            unzip_file(zip_path, images_dir, progress)
            if os.path.exists(zip_path):
                os.remove(zip_path)

    print("All downloads and extractions completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
