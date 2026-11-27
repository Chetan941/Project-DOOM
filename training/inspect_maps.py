
import struct
from pathlib import Path
import re

import vizdoom


ROOT = Path(__file__).resolve().parents[1]

VIZDOOM_DIR = Path(vizdoom.__file__).resolve().parent


def inspect_wad(path):

    with open(path, "rb") as file:

        header = file.read(12)

        magic, count, offset = struct.unpack(
            "<4sii", header
        )

        if magic not in (b"IWAD", b"PWAD"):
            return []

        file.seek(offset)

        maps = []

        for _ in range(count):

            entry = file.read(16)

            _, _, raw_name = struct.unpack(
                "<ii8s", entry
            )

            name = raw_name.rstrip(
                b"\x00"
            ).decode(
                "ascii", errors="ignore"
            )

            if re.fullmatch(
                r"MAP\d{2}|E\dM\d", name
            ):

                maps.append(name)

        return maps


locations = {
    "OFFICIAL MAPS": ROOT / "maps",
    "OFFICIAL TEST FIXTURES": ROOT / "tests" / "fixtures",
    "VIZDOOM PACKAGE": VIZDOOM_DIR,
}

print("\nPROJECT-DOOM MAP INVENTORY")
print("=" * 60)

for category, directory in locations.items():

    print(f"\n{category}")
    print("=" * 60)

    if not directory.exists():

        print("Directory not found")
        continue

    wad_files = sorted(
        directory.rglob("*.wad")
    )

    if not wad_files:

        print("No WAD files found")
        continue

    for path in wad_files:

        try:

            maps = inspect_wad(path)

            print(f"\nFile: {path.name}")

            print(
                f"Size: "
                f"{path.stat().st_size / 1024:.2f} KB"
            )

            print(
                f"Maps: {', '.join(maps) or 'None'}"
            )

            print("Path:", path)

        except Exception as error:

            print(f"Error: {error}")

print("\nMAP INVENTORY COMPLETE")
