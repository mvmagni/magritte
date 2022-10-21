from dataclasses import dataclass

# Holds all relevant processing/cleaning parameters
@dataclass        
class TensorParms:
    __version=0.1
    
    loss: str = 'sparse_categorical_crossentropy'
    optimizer: ... = 'adam'
    metrics: list = None
    validation_split: float = 0.25  
    epochs: int = 25
    batch_size: int = 32
    shuffle: bool = True
    verbose: int = 0 
    
    def __post_init__(self):
        if self.metrics is None:
            #Wasn't defined by user
            metrics = ['accuracy']
        