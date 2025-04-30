# SONIC

## Initializing the Database

The database is initialized using the following command:

```python
from sonicdb import sonic 

db = sonic.Database("database_name.db")
```

This creates a SQLite 

## Models 

The following models make up the database.

### File 

The File model stores the metadata of the audio files. The table below describes the columns, and properties, of the File model.

| Column | Type | Description |
| --- | --- | --- |
| id | Integer | Primary key |

Accessing the File model is done using the following command:

```python
from sonicdb import sonic, models

db = sonic.Database("database_name.db")

file = db.File()
```


### Sensor

### Channel

### Event

### Sample

### Classification