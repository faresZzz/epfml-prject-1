"""
Run the tests provided by the teachers. (replace shell script)
"""
import subprocess
from pathlib import Path


def run_tests():
    test_path = Path('material/grading_tests/test_project1_public.py').resolve().parent
    current_path = Path(__file__).resolve().parent
    command = ["pytest", "--github_link", f'"{current_path}"']
    p = subprocess.call(command, cwd=test_path)
    if p != 0:
        print("Tests failed")
        exit(1)


if __name__ == '__main__':
    run_tests()
