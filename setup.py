from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="veritas",
    version="0.1.0",
    author="Veritas Team",
    author_email="example@example.com",
    description="A RAG-based question answering system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/veritas",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    scripts=[
        "scripts/1-clean_encoding.py",
        "scripts/2-process_json.py",
        "scripts/3-analyze_fulltext.py",
        "scripts/4-chunk_data.py",
        "scripts/5-examine_chunks.py",
        "scripts/6-analyze_chunks.py",
        "scripts/7-index_chunks_parallel.py",
        "scripts/run.py",
        "scripts/cli.py",
        "scripts/build_rag.py",
        "scripts/adapter.py"
    ],
) 