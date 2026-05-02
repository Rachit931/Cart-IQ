<<<<<<< HEAD
import pandas as pd


def time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-based customer-level features
    from transaction-level data.

    Features:
    - CustomerTenure: Total active duration of customer (days)
    - RecencyNormalized: Recency scaled by tenure
      (captures inactivity relative to customer lifetime)

    Notes:
    - Raw recency is NOT returned (already in RFM)
    - No frequency-based features (avoid duplication wtih RFM)

    Parameters:
    - df: cleaned transactional dataframe

    Returns:
=======
import pandas as pd 

def time_based_features(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Build time-based customer-level features
    from transaction-level data. 
    
    Features: 
    - CustomerTenure: Total active duration of customer (days)
    - RecencyNormalized: Recency scaled by tenure 
      (captures inactivity relative to customer lifetime)
    
    Notes: 
    - Raw recency is NOT returned (already in RFM)
    - No frequency-based features (avoid duplication wtih RFM)
    
    Parameters: 
    - df: cleaned transactional dataframe 
    
    Returns: 
>>>>>>> 366aeaf (feat: add time based features)
    - time_features DataFrame
    """

    df = df.copy()
<<<<<<< HEAD
=======

    
>>>>>>> 366aeaf (feat: add time based features)
