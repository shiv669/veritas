from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()
    # Remove comments and empty lines
    requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="veritas",
    version="1.2.0",  # Updated version to match CHANGELOG
    author="Veritas Team",
    author_email="example@example.com",
    description="High-Performance RAG for Apple Silicon with AI Scientist capabilities",
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
    python_requires=">=3.9",  # Updated minimum Python version
    install_requires=requirements,
    scripts=[
        # RAG System Scripts
        "scripts/1-clean_encoding.py",
        "scripts/2-process_json.py",
        "scripts/3-analyze_fulltext.py",
        "scripts/4-chunk_data.py",
        "scripts/5-examine_chunks.py",
        "scripts/6-analyze_chunks.py",
        "scripts/7-index_chunks_parallel.py",
        "scripts/run.py",  # Unified entry point
        "scripts/cli.py",
        "scripts/build_rag.py",
        "scripts/adapter.py",
        
        # AI Scientist Scripts
        "src/veritas/ai_scientist/run_interface.py",
        "src/veritas/ai_scientist/run_scientist.py",
        "src/veritas/ai_scientist/test_simple.py",
        "src/veritas/ai_scientist/test_system.py",
        "src/veritas/ai_scientist/test_all.sh",
    ],
    entry_points={
        'console_scripts': [
            'veritas=scripts.run:main',
            'veritas-ai-scientist=src.veritas.ai_scientist.run_interface:main',
        ],
    },
) 