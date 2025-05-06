```{toctree}
---
hidden: true
maxdepth: 3
includehidden:
caption: User Guide
---

getting_started
sonic
audio
sonicdb
Changelog <https://github.com/dkadyrov/sonicdb/releases>
```

## About

The Sound Organization and Network Integration for Collection/Collaboration Database (SONICDB) Python package leverages relational database management systems to store and manage acoustic data and its associated metadata.

It was developed to address the lack of a standardized method of organizing audio files for research purposes, especially in the context of algorithm development and machine learning.

The package is designed to be flexible and extendable, allowing users to retrofit it to their use cases while maintaining foundational models and methods for managing audio data. It uses widely used standard libraries including SQLAlchemy, Pandas, Numpy, and Librosa. Currently it emphasizes using SQLite as the database backend, but it can be extended to use other relational database management systems.

This package has been used to manage the following datasets: 

- [Stored Product Insect Detection Database (SPIDB)](https://github.com/dkadyrov/spidb)
- Drone Acoustic Dataset (DAD)