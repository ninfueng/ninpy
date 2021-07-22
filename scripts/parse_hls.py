"""A script to parse Vivado High-level Synthesis reports to csv file."""
import glob
import os
import sys
from typing import List
from xml.etree import ElementTree as ET


def parse_hls_xml(xml_dir: str):
    """Parse a xml file and collects all necessary information."""
    assert isinstance(xml_dir, str)
    tree = ET.parse(xml_dir)
    root = tree.getroot()

    return


if __name__ == "__main__":
    try:
        top_fn, xml_dir, hls_type = sys.argv[1], sys.argv[2], sys.argv[3]
    except IndexError:
        raise IndexError(
            "Please an arg0, arg1, and arg2 as shown in"
            "`python parse_hls_xml.py arg0 arg1 arg2`."
        )

    xml_dir = os.path.expanduser(xml_dir)
    xml_dirs = glob.glob(os.path.join(xml_dir, "*.xml"), recursive=True)
    print(xml_dirs)

    # TODO: finish this file.
    # solution name
    # syn, impl, other?
    # type?
