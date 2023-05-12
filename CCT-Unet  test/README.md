# Test for CCT-Unet

## Perpare Test Data

Place the data to be tested in the following directory path::

```
/data/
├── segdata/
    ├── img/
        ├── [test_data].png
        ...
    ├── ann/
        ├── [test_data].png
        ...
```

## Testing  Phase
Download the pre-trained model to the CCT-Unet test directory, and you could run  `python model_test.py` for testing your data.

