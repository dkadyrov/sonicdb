---
hide-toc: true
---

# SONICDB

## About

The Sound Organization and Network Integration for Collection/Collaboration (SONIC) Python package leverages relational database management systems to store and manage acoustic data and its associated metadata.

It was developed to address the lack of a standardized method of organizing audio files for research purposes, especially in the context of algorithm development and machine learning.

The package is designed to be flexible and extendable, allowing users to retrofit it to their use cases while maintaining foundational models and methods for managing audio data. It uses widely used standard libraries including SQLAlchemy, Pandas, Numpy, and Librosa. Currently it emphasizes using SQLite as the database backend, but it can be extended to use other relational database management systems.

## Usage

### Requirements

The SONICDB Python package requires Python 3.12 or later. It also depends on the following packages:

- SQLAlchemy
- Pandas
- Numpy
- Librosa
- Pydub
- SQLAlchemy-Utils
- Soundfile
- Scipy
- Matplotlib

### Installation

SONICDB is available on PyPi and can be installed using pip:

```bash
pip install sonicdb
```

It can also be installed from the source using the following command:

```bash
pip install git+https://github.com/dkadyrov/sonicdb.git
```

### Documentation

The documentation for the SONIC Python package can be found [here](/guide/guide).

```{toctree}
:hidden:

guide
audio
Changelog <https://github.com/dkadyrov/sonicdb/releases>
```
